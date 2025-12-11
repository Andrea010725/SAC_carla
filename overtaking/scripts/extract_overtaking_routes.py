#!/usr/bin/env python3
"""
从 bench2drive220.xml 中提取 Overtaking 能力相关的路线
生成独立的 overtaking.xml 文件
"""

import xml.etree.ElementTree as ET
import sys
import os

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

def extract_overtaking_routes(input_xml, output_xml):
    """
    提取Overtaking相关的路线

    Args:
        input_xml: 输入的完整XML文件路径
        output_xml: 输出的Overtaking专用XML文件路径
    """
    print(f"📖 读取文件: {input_xml}")
    tree = ET.parse(input_xml)
    root = tree.getroot()

    # 创建新的根节点
    new_root = ET.Element('routes')

    extracted_count = 0
    route_ids = []

    # 遍历所有路线
    for route in root.findall('route'):
        # 查找该路线的scenario类型
        scenarios = route.find('scenarios')
        if scenarios is not None:
            scenario = scenarios.find('scenario')
            if scenario is not None:
                scenario_type = scenario.get('type')

                # 如果是Overtaking相关的场景，则提取
                if scenario_type in OVERTAKING_SCENARIOS:
                    new_root.append(route)
                    extracted_count += 1
                    route_ids.append(route.get('id'))
                    print(f"   ✅ 提取路线 {route.get('id')}: {scenario_type}")

    # 保存为新的XML文件
    new_tree = ET.ElementTree(new_root)

    # 直接保存（不使用indent，兼容Python 3.7）
    new_tree.write(output_xml, encoding='utf-8', xml_declaration=True)

    print(f"\n✅ 提取完成!")
    print(f"   总共提取: {extracted_count} 条路线")
    print(f"   输出文件: {output_xml}")
    print(f"\n📋 路线ID列表:")
    print(f"   {','.join(route_ids)}")

    # 统计各场景类型的数量
    print(f"\n📊 场景类型统计:")
    for scenario_type in OVERTAKING_SCENARIOS:
        count = sum(1 for route in new_root.findall('route')
                   if route.find('.//scenario').get('type') == scenario_type)
        if count > 0:
            print(f"   {scenario_type}: {count} 条")

    return route_ids

def main():
    # 设置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_xml = os.path.join(project_root, '../leaderboard/data/bench2drive220.xml')
    output_xml = os.path.join(project_root, 'data/overtaking.xml')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_xml), exist_ok=True)

    print("=" * 70)
    print("🚗 Overtaking 路线提取工具")
    print("=" * 70)
    print()

    # 提取路线
    route_ids = extract_overtaking_routes(input_xml, output_xml)

    # 保存路线ID列表
    route_ids_file = os.path.join(project_root, 'data/overtaking_route_ids.txt')
    with open(route_ids_file, 'w') as f:
        f.write(','.join(route_ids))

    print(f"\n💾 路线ID已保存到: {route_ids_file}")
    print()
    print("=" * 70)

if __name__ == '__main__':
    main()
