"""
XML 场景配置解析器

从 ScenarioRunner 的 XML 文件中读取预定义的场景位置
"""

import os
import xml.etree.ElementTree as ET
import random
from typing import Optional, Dict, List, Tuple
import carla


class ScenarioXMLParser:
    """解析 ScenarioRunner XML 配置文件"""

    def __init__(self, xml_dir: str = None):
        """
        初始化 XML 解析器

        Args:
            xml_dir: XML 文件目录，默认为 b2drive scenario_runner 目录
        """
        if xml_dir is None:
            # 默认路径
            xml_dir = "/home/ajifang/b2drive/scenario_runner/srunner/examples"

        self.xml_dir = xml_dir
        self.scenarios_cache = {}  # 缓存解析结果

    def parse_xml_file(self, xml_file: str) -> List[Dict]:
        """
        解析单个 XML 文件

        Args:
            xml_file: XML 文件路径

        Returns:
            List[Dict]: 场景配置列表
        """
        if not os.path.exists(xml_file):
            print(f"[XMLParser] ⚠️ XML 文件不存在: {xml_file}")
            return []

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            scenarios = []
            for scenario in root.findall('scenario'):
                scenario_data = {
                    'name': scenario.get('name'),
                    'type': scenario.get('type'),
                    'town': scenario.get('town'),
                }

                # 解析 ego_vehicle
                ego = scenario.find('ego_vehicle')
                if ego is not None:
                    scenario_data['ego_vehicle'] = {
                        'x': float(ego.get('x', 0)),
                        'y': float(ego.get('y', 0)),
                        'z': float(ego.get('z', 0)),
                        'yaw': float(ego.get('yaw', 0)),
                        'model': ego.get('model', 'vehicle.tesla.model3'),
                    }

                # 解析 other_actor
                other_actors = []
                for actor in scenario.findall('other_actor'):
                    other_actors.append({
                        'x': float(actor.get('x', 0)),
                        'y': float(actor.get('y', 0)),
                        'z': float(actor.get('z', 0)),
                        'yaw': float(actor.get('yaw', 0)),
                        'model': actor.get('model', 'vehicle.tesla.model3'),
                    })
                if other_actors:
                    scenario_data['other_actors'] = other_actors

                # 解析其他参数
                for param in scenario:
                    if param.tag not in ['ego_vehicle', 'other_actor', 'weather']:
                        scenario_data[param.tag] = param.get('value')

                scenarios.append(scenario_data)

            return scenarios

        except Exception as e:
            print(f"[XMLParser] ❌ 解析 XML 失败: {e}")
            return []

    def get_scenarios_for_type(self, scenario_type: str, town: str = None) -> List[Dict]:
        """
        获取指定类型的场景配置

        Args:
            scenario_type: 场景类型（如 "CutIn", "VehicleOpensDoorTwoWays"）
            town: 地图名称（如 "Town05"），如果为 None 则返回所有地图的场景

        Returns:
            List[Dict]: 匹配的场景配置列表
        """
        # 构建 XML 文件路径
        xml_file = os.path.join(self.xml_dir, f"{scenario_type}.xml")

        # 如果文件不存在，尝试其他可能的文件名
        if not os.path.exists(xml_file):
            # 尝试查找包含该类型的 XML 文件
            for filename in os.listdir(self.xml_dir):
                if filename.endswith('.xml'):
                    full_path = os.path.join(self.xml_dir, filename)
                    scenarios = self.parse_xml_file(full_path)
                    for scenario in scenarios:
                        if scenario.get('type') == scenario_type:
                            xml_file = full_path
                            break

        # 解析 XML 文件
        if xml_file not in self.scenarios_cache:
            self.scenarios_cache[xml_file] = self.parse_xml_file(xml_file)

        scenarios = self.scenarios_cache[xml_file]

        # 过滤地图
        if town:
            scenarios = [s for s in scenarios if s.get('town') == town]

        return scenarios

    def get_random_scenario_config(
        self,
        scenario_type: str,
        town: str = None
    ) -> Optional[Dict]:
        """
        随机获取一个场景配置

        Args:
            scenario_type: 场景类型
            town: 地图名称

        Returns:
            Dict: 场景配置，如果没有找到返回 None
        """
        scenarios = self.get_scenarios_for_type(scenario_type, town)

        if not scenarios:
            return None

        return random.choice(scenarios)

    def create_transform_from_config(self, config: Dict) -> Optional[carla.Transform]:
        """
        从配置创建 Transform

        Args:
            config: 包含 x, y, z, yaw 的配置字典

        Returns:
            carla.Transform: CARLA Transform 对象
        """
        if 'ego_vehicle' in config:
            ego = config['ego_vehicle']
            location = carla.Location(
                x=float(ego['x']),
                y=float(ego['y']),
                z=float(ego['z'])
            )
            rotation = carla.Rotation(yaw=float(ego['yaw']))
            return carla.Transform(location, rotation)

        return None


# 全局解析器实例
_xml_parser = None


def get_xml_parser() -> ScenarioXMLParser:
    """获取全局 XML 解析器实例"""
    global _xml_parser
    if _xml_parser is None:
        _xml_parser = ScenarioXMLParser()
    return _xml_parser


# 场景类型映射（我们的场景名 -> ScenarioRunner 场景类型）
SCENARIO_TYPE_MAPPING = {
    "vehicle_opens_door": "VehicleOpensDoorTwoWays",
    "cut_in": "CutIn",
    "pedestrian_crossing": "DynamicObjectCrossing",  # 可以用行人穿越场景的位置
    "parking_exit": "VehicleOpensDoorTwoWays",  # 可以复用车门场景的位置
}


def get_predefined_spawn_for_scenario(
    scenario_name: str,
    town: str = "Town05"
) -> Optional[carla.Transform]:
    """
    获取场景的预定义 spawn 位置

    Args:
        scenario_name: 场景名称（如 "vehicle_opens_door"）
        town: 地图名称

    Returns:
        carla.Transform: spawn 位置，如果没有找到返回 None
    """
    # 获取对应的 ScenarioRunner 场景类型
    scenario_type = SCENARIO_TYPE_MAPPING.get(scenario_name)

    if not scenario_type:
        return None

    # 获取 XML 解析器
    parser = get_xml_parser()

    # 获取随机配置
    config = parser.get_random_scenario_config(scenario_type, town)

    if not config:
        print(f"[XMLParser] ⚠️ 未找到 {scenario_name} 在 {town} 的预定义位置")
        return None

    # 创建 Transform
    transform = parser.create_transform_from_config(config)

    if transform:
        print(f"[XMLParser] ✅ 使用预定义位置: {scenario_name} @ {town}")
        print(f"            位置: ({transform.location.x:.1f}, {transform.location.y:.1f}, {transform.location.z:.1f})")
        print(f"            朝向: Yaw={transform.rotation.yaw:.1f}°")

    return transform
