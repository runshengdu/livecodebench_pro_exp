from abc import ABC, abstractmethod
from typing import Self
import docker
import docker.errors
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import logging
import requests
import time
import os
import json
from enum import Enum
from zipfile import ZipFile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

SIGNAL_NAMES = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 6: "SIGABRT",
    7: "SIGBUS", 8: "SIGFPE", 9: "SIGKILL", 10: "SIGUSR1",
    11: "SIGSEGV", 12: "SIGUSR2", 13: "SIGPIPE", 14: "SIGALRM",
    15: "SIGTERM", 16: "SIGSTKFLT", 17: "SIGCHLD", 18: "SIGCONT",
    19: "SIGSTOP", 20: "SIGTSTP", 21: "SIGTTIN", 22: "SIGTTOU",
    23: "SIGURG", 24: "SIGXCPU", 25: "SIGXFSZ", 26: "SIGVTALRM",
    27: "SIGPROF", 28: "SIGWINCH", 29: "SIGIO", 30: "SIGPWR",
    31: "SIGSYS", 34: "SIGRTMIN", 35: "SIGRTMIN+1", 36: "SIGRTMIN+2",
    37: "SIGRTMIN+3", 38: "SIGRTMIN+4", 39: "SIGRTMIN+5", 40: "SIGRTMIN+6",
    41: "SIGRTMIN+7", 42: "SIGRTMIN+8", 43: "SIGRTMIN+9", 44: "SIGRTMIN+10",
    45: "SIGRTMIN+11", 46: "SIGRTMIN+12", 47: "SIGRTMIN+13", 48: "SIGRTMIN+14",
    49: "SIGRTMIN+15", 50: "SIGRTMAX-14", 51: "SIGRTMAX-13", 52: "SIGRTMAX-12",
    53: "SIGRTMAX-11", 54: "SIGRTMAX-10", 55: "SIGRTMAX-9", 56: "SIGRTMAX-8",
    57: "SIGRTMAX-7", 58: "SIGRTMAX-6", 59: "SIGRTMAX-5", 60: "SIGRTMAX-4",
    61: "SIGRTMAX-3", 62: "SIGRTMAX-2", 63: "SIGRTMAX-1", 64: "SIGRTMAX"
}

def create_session_with_retry(retries=5, backoff_factor=0.5, pool_connections=10, pool_maxsize=10):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

class SupportedLanguage(Enum):
    CPP = "cpp"
    PYTHON3 = "python3"
    PYPY3 = "pypy3"

class Judge(ABC):
    @abstractmethod
    def __enter__(self) -> Self:
        pass
    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    @abstractmethod
    def submit(self, problem_id: str, language: SupportedLanguage, code: str) -> int:
        pass
    @abstractmethod
    def get_result(self, submission_id: int) -> str:
        pass
    @abstractmethod
    def get_statistics(self) -> dict:
        pass
    @abstractmethod
    def print_statistics(self):
        pass

class ProblemNotFoundError(Exception):
    pass

class LightCPVerifierJudge(Judge):
    IMAGE_NAME = "lightcpverifier"
    CONTAINER_NAME = "lightcpverifier"
    REPO_DIR = os.path.realpath("LightCPVerifier")
    PROBLEMS_DIR = os.path.join(REPO_DIR, "problems")
    DATASET_HF_REPO = "QAQAQAQAQ/LiveCodeBench-Pro-Testcase"

    def __init__(self, worker: int = 4, memory_limit: str = "32g", cpu_quota: int = 1600000, 
                 submit_timeout: int = 120, result_timeout: int = 30, 
                 execution_timeout: int = 120, max_retries: int = 5):
        self.worker = worker
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.submit_timeout = submit_timeout
        self.result_timeout = result_timeout
        self.execution_timeout = execution_timeout
        self.max_retries = max_retries
        self.session = None
        self._downloaded_problems = set()
        self._stats = {
            "total_submissions": 0,
            "successful_submissions": 0,
            "failed_submissions": 0,
            "total_result_checks": 0,
            "successful_result_checks": 0,
            "failed_result_checks": 0,
            "problems_downloaded": 0,
            "download_cache_hits": 0
        }

    def __enter__(self):
        self.docker_client = docker.from_env()
        self.session = create_session_with_retry()
        self._build_image()
        self._start_container()
        self._ensure_connection()
        return self

    def _build_image(self):
        try:
            self.docker_client.images.get(self.IMAGE_NAME)
            logger.info(f"Image '{self.IMAGE_NAME}' found locally.")
        except docker.errors.ImageNotFound:
            logger.warning(f"Image '{self.IMAGE_NAME}' not found. Building it now...")
            self.docker_client.images.build(
                path=self.REPO_DIR,
                tag=self.IMAGE_NAME,
            )
            logger.info(f"Image '{self.IMAGE_NAME}' built successfully.")

    def _start_container(self):
        try:
            existing_container = self.docker_client.containers.get(self.CONTAINER_NAME)
            logger.info(f"Container '{self.CONTAINER_NAME}' already exists. Removing it...")
            existing_container.stop()
            existing_container.remove()
            logger.info(f"Container '{self.CONTAINER_NAME}' removed.")
        except docker.errors.NotFound: pass
        
        os.makedirs(self.PROBLEMS_DIR, exist_ok=True)
        
        self.container = self.docker_client.containers.run(
            image=self.IMAGE_NAME,
            name=self.CONTAINER_NAME,
            privileged=True,
            detach=True,
            shm_size="32g",
            mem_limit=self.memory_limit,
            memswap_limit=self.memory_limit,
            cpu_period=100000,
            cpu_quota=self.cpu_quota,
            environment={
                "JUDGE_WORKERS": str(self.worker),
                "GJ_PARALLELISM": str(self.worker),
                "CODE_EXECUTION_TIMEOUT": str(self.execution_timeout),
                "RUN_HIDEOUT": "1",
                "GOMEMLIMIT": "16GiB",
            },
            volumes=[
                f"{self.PROBLEMS_DIR}:/app/problems",
                f"{os.path.join(self.REPO_DIR, "submissions")}:/app/submissions",
                f"{os.path.join(self.REPO_DIR, "data")}:/app/data",
            ],
            ports={"8081/tcp": None},
            restart_policy={"Name": "on-failure", "MaximumRetryCount": 3},
            cap_add=["SYS_ADMIN"],
            security_opt=["seccomp=unconfined"],
            pids_limit=4096,
            oom_kill_disable=False,
            ulimits=[docker.types.Ulimit(name='nofile', soft=65536, hard=65536)],
        )
        
        while self.container.status != "running":
            time.sleep(1)
            self.container.reload()
        
        port = self.container.ports['8081/tcp'][0]['HostPort']
        self.base_url = f"http://localhost:{port}"
        logger.info(f"Container '{self.CONTAINER_NAME}' started with {self.memory_limit} memory, {self.cpu_quota//100000} CPUs, listening on port {port}.")

    def _check_connection(self):
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            logger.debug(f"Health check successful: {response.status_code}")
            return True
        except requests.Timeout:
            logger.error(f"Health check timed out to {self.base_url}")
            return False
        except requests.ConnectionError as e:
            logger.error(f"Health check connection failed to {self.base_url}: {e}")
            return False
        except requests.RequestException as e:
            logger.error(f"Health check failed to {self.base_url}: {e}")
            return False

    def _ensure_connection(self):
        logger.info("Checking connection to judge service...")
        for _ in range(30):
            if self._check_connection():
                logger.info("Connection to judge service established.")
                return
            time.sleep(2)
        raise RuntimeError("Failed to connect to judge service after multiple attempts.")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.container:
            try:
                logger.info(f"Stopping and removing container '{self.CONTAINER_NAME}'...")
                self.container.stop()
                self.container.remove()
                logger.info(f"Container '{self.CONTAINER_NAME}' stopped and removed.")
            except Exception as e:
                logger.error(f"Error stopping/removing container '{self.CONTAINER_NAME}': {e}")
        self.session.close()
        self.print_statistics()

    def _ensure_data_downloaded(self, problem_id: str):
        if problem_id in self._downloaded_problems:
            self._stats["download_cache_hits"] += 1
            return
        
        problem_dir = os.path.join(self.PROBLEMS_DIR, problem_id)
        if os.path.exists(os.path.join(problem_dir, "config.yaml")):
            self._downloaded_problems.add(problem_id)
            self._stats["problems_downloaded"] += 1
            return
            
        logger.info(f"Downloading data for problem '{problem_id}'...")
        try:
            zip_path = hf_hub_download(
                repo_id=self.DATASET_HF_REPO,
                filename=f"{problem_id}.zip",
                repo_type="dataset",
            )
        except EntryNotFoundError:
            logger.error(f"Problem '{problem_id}' not found in dataset repository '{self.DATASET_HF_REPO}'")
            raise ProblemNotFoundError(f"Problem '{problem_id}' not found in dataset repository.")
        except Exception as e:
            logger.error(f"Failed to download problem '{problem_id}': {e}")
            raise RuntimeError(f"Failed to download problem '{problem_id}': {e}")
            
        os.makedirs(problem_dir, exist_ok=True)
        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(problem_dir)
            self._downloaded_problems.add(problem_id)
            self._stats["problems_downloaded"] += 1
            logger.info(f"Data for problem '{problem_id}' downloaded and extracted successfully.")
        except Exception as e:
            logger.error(f"Failed to extract problem '{problem_id}': {e}")
            raise RuntimeError(f"Failed to extract problem '{problem_id}': {e}")

    def submit(self, problem_id: str, language: SupportedLanguage, code: str) -> int:
        self._ensure_data_downloaded(problem_id)
        self._stats["total_submissions"] += 1
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/submit",
                    json={
                        "pid": problem_id,
                        "lang": language.value,
                        "code": code
                    },
                    timeout=self.submit_timeout
                )
                if response.status_code != 200:
                    logger.warning(f"Submit attempt {attempt + 1}/{self.max_retries} failed with status {response.status_code}")
                    if attempt < self.max_retries - 1:
                        time.sleep(min(2 ** attempt, 10))
                    continue
                response.raise_for_status()
                self._stats["successful_submissions"] += 1
                return response.json()["sid"]
            except requests.Timeout as e:
                logger.warning(f"Submit attempt {attempt + 1}/{self.max_retries} timed out: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
            except requests.ConnectionError as e:
                logger.warning(f"Submit attempt {attempt + 1}/{self.max_retries} connection error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
            except requests.RequestException as e:
                logger.warning(f"Submit attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
        
        self._stats["failed_submissions"] += 1
        raise RuntimeError(f"Failed to submit after {self.max_retries} attempts")

    def get_result(self, submission_id: int) -> str:
        self._stats["total_result_checks"] += 1
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}/result/{submission_id}",
                    timeout=self.result_timeout
                )
                if response.status_code == 404:
                    return "Judging"
                response.raise_for_status()
                result = response.json()
                
                logger.debug(f"完整响应 (submission_id={submission_id}): {json.dumps(result, indent=2)}")
                
                if result["status"] == "queued":
                    return "Judging"
                if result["status"] == "error":
                    error_details = result.get("error", "Unknown error")
                    logger.error(f"Judge error for submission {submission_id}: {error_details}")
                    self._stats["failed_result_checks"] += 1
                    return f"Judge Failed: {error_details}"
                if result["result"] == "Signalled":
                    signal_num = result.get("signal", result.get("error", result.get("status", None)))
                    logger.error(f"Signalled - 原始响应: {json.dumps(result, indent=2)}")
                    if signal_num is not None:
                        try:
                            signal_num = int(signal_num)
                            signal_name = SIGNAL_NAMES.get(signal_num, f"Signal-{signal_num}")
                            self._stats["failed_result_checks"] += 1
                            return f"Signalled ({signal_name}, {signal_num})"
                        except (ValueError, TypeError):
                            self._stats["failed_result_checks"] += 1
                            return f"Signalled ({signal_num})"
                    self._stats["failed_result_checks"] += 1
                    return "Signalled (unknown signal)"
                self._stats["successful_result_checks"] += 1
                return result["result"]
            except requests.Timeout as e:
                logger.warning(f"Get result attempt {attempt + 1}/{self.max_retries} timed out for submission {submission_id}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
            except requests.ConnectionError as e:
                logger.warning(f"Get result attempt {attempt + 1}/{self.max_retries} connection error for submission {submission_id}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
            except requests.RequestException as e:
                logger.warning(f"Get result attempt {attempt + 1}/{self.max_retries} failed for submission {submission_id}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
        self._stats["failed_result_checks"] += 1
        return "Judge Failed: Network error after retries"

    def get_statistics(self) -> dict:
        return self._stats.copy()

    def print_statistics(self):
        stats = self.get_statistics()
        logger.info("=" * 50)
        logger.info("Judge System Statistics")
        logger.info("=" * 50)
        logger.info(f"Total Submissions: {stats['total_submissions']}")
        logger.info(f"  Successful: {stats['successful_submissions']}")
        logger.info(f"  Failed: {stats['failed_submissions']}")
        logger.info(f"Total Result Checks: {stats['total_result_checks']}")
        logger.info(f"  Successful: {stats['successful_result_checks']}")
        logger.info(f"  Failed: {stats['failed_result_checks']}")
        logger.info(f"Problems Downloaded: {stats['problems_downloaded']}")
        logger.info(f"Download Cache Hits: {stats['download_cache_hits']}")
        logger.info("=" * 50)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with LightCPVerifierJudge(worker=2) as judge:
        sid = judge.submit("2000A", SupportedLanguage.CPP, "#include <bits/stdc++.h>\nusing namespace std;\n\n\nint main()\n{   int t;\ncin>>t;\nwhile(t--){\n    string s;\n    cin>>s;\n    if(s[0]=='1'&&s[1]=='0'&&s[2]!='0'&&(!(s[2]=='1')||s.length()>3)&&s.length()>2){cout<<\"YES\"<<endl; }\n    else cout<<\"NO\"<<endl;\n\n}\n\n    return 0;\n}\n\n")
        print(f"Submitted with ID: {sid}")
        while True:
            result = judge.get_result(sid)
            print(f"Result: {result}")
            if result != "Judging":
                break
            time.sleep(2)
