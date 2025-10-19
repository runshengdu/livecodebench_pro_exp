#!/usr/bin/env python3
import json

result_file = "/Users/miao/Desktop/GBAI/Resources/Test/LiveCodeBench-Pro/result/benchmark_result_google_gemini-3-flash-preview_20260114_132640.json"

with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

difficulty_stats = {}

for item in data:
    difficulty = item.get('difficulty', 'unknown')
    judge_result = item.get('judge_result', '')
    
    if difficulty not in difficulty_stats:
        difficulty_stats[difficulty] = {
            'total': 0,
            'accepted': 0,
            'failed': 0,
            'tasks': []
        }
    
    difficulty_stats[difficulty]['total'] += 1
    difficulty_stats[difficulty]['tasks'].append({
        'problem_id': item['problem_id'],
        'problem_title': item['problem_title'],
        'judge_result': judge_result
    })
    
    if judge_result == 'Accepted':
        difficulty_stats[difficulty]['accepted'] += 1
    else:
        difficulty_stats[difficulty]['failed'] += 1

print("=" * 80)
print("Difficulty-wise Accuracy Statistics")
print("=" * 80)
print(f"{'Difficulty':<15} | {'Total':<8} | {'Accepted':<10} | {'Failed':<8} | {'Accuracy':<10}")
print("-" * 80)

for difficulty in sorted(difficulty_stats.keys()):
    stats = difficulty_stats[difficulty]
    accuracy = (stats['accepted'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"{difficulty:<15} | {stats['total']:<8} | {stats['accepted']:<10} | {stats['failed']:<8} | {accuracy:.2f}%")

print("=" * 80)
print(f"{'TOTAL':<15} | {len(data):<8} | {sum(s['accepted'] for s in difficulty_stats.values()):<10} | {sum(s['failed'] for s in difficulty_stats.values()):<8} | {sum(s['accepted'] for s in difficulty_stats.values()) / len(data) * 100:.2f}%")
print("=" * 80)

print("\nDetailed breakdown by difficulty:")
print("=" * 80)
for difficulty in sorted(difficulty_stats.keys()):
    stats = difficulty_stats[difficulty]
    accuracy = (stats['accepted'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"\n{difficulty.upper()} (Accuracy: {accuracy:.2f}%)")
    print("-" * 80)
    for task in stats['tasks']:
        status = "✓" if task['judge_result'] == 'Accepted' else "✗"
        print(f"{status} {task['problem_id']:10s} | {task['problem_title']:50s} | {task['judge_result']}")
