#!/usr/bin/env python3
"""
Overtaking能力测试结果分析脚本
分析每个场景类型的表现，生成详细报告
"""

import json
import sys
import os
from typing import Dict, List

# Overtaking能力包含的场景类型
OVERTAKING_SCENARIOS = [
    'Accident',
    'AccidentTwoWays',
    'ConstructionObstacle',
    'ConstructionObstacleTwoWays',
    'HazardAtSideLaneTwoWays',
    'HazardAtSideLane',
    'ParkedObstacleTwoWays',
    'ParkedObstacle',
    'VehicleOpensDoorTwoWays'
]

# 场景类型中文名
SCENARIO_NAMES_CN = {
    'Accident': '事故场景',
    'AccidentTwoWays': '双向道路事故',
    'ConstructionObstacle': '施工障碍',
    'ConstructionObstacleTwoWays': '双向道路施工',
    'HazardAtSideLaneTwoWays': '双向道路侧方危险',
    'HazardAtSideLane': '侧方危险',
    'ParkedObstacleTwoWays': '双向道路停车障碍',
    'ParkedObstacle': '停车障碍',
    'VehicleOpensDoorTwoWays': '双向道路开车门'
}

def get_infraction_status(record: Dict) -> bool:
    """
    检查是否有违规（除了最小速度）

    Returns:
        True: 有违规
        False: 无违规
    """
    for infraction, value in record['infractions'].items():
        if infraction == "min_speed_infractions":
            continue
        elif len(value) > 0:
            return True
    return False

def analyze_results(result_file: str) -> Dict:
    """
    分析Overtaking测试结果

    Args:
        result_file: 结果JSON文件路径

    Returns:
        分析结果字典
    """
    print("=" * 70)
    print("📊 Overtaking 能力测试结果分析")
    print("=" * 70)
    print()

    # 读取结果文件
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return {}

    with open(result_file, 'r') as f:
        data = json.load(f)

    records = data.get('records', [])

    if not records:
        print("❌ 结果文件中没有记录")
        return {}

    print(f"📋 总路线数: {len(records)}")
    print()

    # 统计各场景类型的表现
    scenario_stats = {}
    for scenario in OVERTAKING_SCENARIOS:
        scenario_stats[scenario] = {
            'total': 0,
            'success': 0,
            'completed': 0,
            'failed': 0,
            'route_ids': []
        }

    # 分析每条记录
    for record in records:
        # 从route_id中提取场景类型
        # route_id格式: RouteScenario_1773
        route_id = record['route_id']
        status = record['status']

        # 查找该路线属于哪个场景
        # 需要通过其他方式获取场景类型，这里暂时从meta中获取
        # 如果没有meta信息，需要从XML中匹配
        scenario_type = record.get('meta', {}).get('scenario_type', 'Unknown')

        if scenario_type in scenario_stats:
            scenario_stats[scenario_type]['total'] += 1
            scenario_stats[scenario_type]['route_ids'].append(route_id)

            if status == 'Completed' or status == 'Perfect':
                scenario_stats[scenario_type]['completed'] += 1

                # 检查是否有违规
                if not get_infraction_status(record):
                    scenario_stats[scenario_type]['success'] += 1
            else:
                scenario_stats[scenario_type]['failed'] += 1

    # 打印详细统计
    print("=" * 70)
    print("📈 各场景类型表现")
    print("=" * 70)
    print()

    total_success = 0
    total_routes = 0

    for scenario in OVERTAKING_SCENARIOS:
        stats = scenario_stats[scenario]
        cn_name = SCENARIO_NAMES_CN.get(scenario, scenario)

        if stats['total'] > 0:
            success_rate = stats['success'] / stats['total'] * 100
            completion_rate = stats['completed'] / stats['total'] * 100

            total_success += stats['success']
            total_routes += stats['total']

            print(f"🎯 {scenario} ({cn_name})")
            print(f"   总数: {stats['total']} 条")
            print(f"   成功: {stats['success']} 条 ({success_rate:.1f}%)")
            print(f"   完成: {stats['completed']} 条 ({completion_rate:.1f}%)")
            print(f"   失败: {stats['failed']} 条")

            # 显示成功率等级
            if success_rate >= 90:
                print(f"   评级: ⭐⭐⭐ 优秀")
            elif success_rate >= 70:
                print(f"   评级: ⭐⭐ 良好")
            elif success_rate >= 50:
                print(f"   评级: ⭐ 及格")
            else:
                print(f"   评级: ❌ 需改进")
            print()

    # 计算总体得分
    print("=" * 70)
    print("🏆 总体表现")
    print("=" * 70)
    print()

    overall_score = total_success / total_routes * 100 if total_routes > 0 else 0

    print(f"总路线数: {total_routes}")
    print(f"成功路线: {total_success}")
    print(f"Overtaking能力得分: {overall_score:.2f}%")
    print()

    if overall_score >= 90:
        print("🎉 评级: ⭐⭐⭐ 优秀 - Overtaking能力非常强!")
    elif overall_score >= 70:
        print("👍 评级: ⭐⭐ 良好 - Overtaking能力较好")
    elif overall_score >= 50:
        print("✅ 评级: ⭐ 及格 - Overtaking能力基本合格")
    else:
        print("⚠️  评级: ❌ 需改进 - Overtaking能力需要提升")

    print()

    # 保存详细分析结果
    analysis_result = {
        'ability': 'Overtaking',
        'overall_score': overall_score,
        'total_routes': total_routes,
        'success_routes': total_success,
        'scenario_stats': scenario_stats,
        'summary': {
            'excellent': sum(1 for s in scenario_stats.values()
                           if s['total'] > 0 and s['success']/s['total'] >= 0.9),
            'good': sum(1 for s in scenario_stats.values()
                       if s['total'] > 0 and 0.7 <= s['success']/s['total'] < 0.9),
            'pass': sum(1 for s in scenario_stats.values()
                       if s['total'] > 0 and 0.5 <= s['success']/s['total'] < 0.7),
            'need_improve': sum(1 for s in scenario_stats.values()
                               if s['total'] > 0 and s['success']/s['total'] < 0.5)
        }
    }

    return analysis_result

def main():
    # 设置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    overtaking_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(overtaking_dir, 'results')

    result_file = os.path.join(results_dir, 'overtaking_results.json')

    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        print(f"   请先运行测试脚本: {overtaking_dir}/scripts/test_overtaking.sh")
        sys.exit(1)

    # 分析结果
    analysis = analyze_results(result_file)

    if analysis:
        # 保存分析报告
        analysis_file = os.path.join(results_dir, 'overtaking_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print("=" * 70)
        print(f"💾 分析报告已保存: {analysis_file}")
        print("=" * 70)
        print()

if __name__ == '__main__':
    main()
