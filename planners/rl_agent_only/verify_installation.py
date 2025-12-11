#!/usr/bin/env python3
"""
验证 rl_agent_only 文件夹的完整性
检查所有必需的文件是否存在，以及导入是否正常
"""
import os
import sys

def check_file_exists(filepath, required=True):
    """检查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌" if required else "⚠️ ")
    required_text = "必需" if required else "可选"
    print(f"{status} [{required_text}] {filepath}")
    return exists

def check_directory_structure():
    """检查目录结构"""
    print("=" * 80)
    print("📁 检查目录结构")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    required_files = [
        # 核心文件
        "__init__.py",
        "utils.py",
        "README.md",
        "requirements.txt",

        # agents模块
        "agents/__init__.py",
        "agents/ppo.py",
        "agents/agents.py",

        # networks模块
        "networks/__init__.py",
        "networks/networks.py",
        "networks/architectures.py",

        # parameters模块
        "parameters/__init__.py",
        "parameters/parameters.py",

        # augmentations模块
        "augmentations/__init__.py",
        "augmentations/augmentations.py",
    ]

    optional_files = [
        "example_usage.py",
        "FILE_MANIFEST.md",
        "agents/ppo_new.py",
        "agents/agents_new.py",
        "augmentations/simclr.py",
    ]

    print("\n必需文件:")
    all_required_exist = True
    for filepath in required_files:
        full_path = os.path.join(base_dir, filepath)
        if not check_file_exists(full_path, required=True):
            all_required_exist = False

    print("\n可选文件:")
    for filepath in optional_files:
        full_path = os.path.join(base_dir, filepath)
        check_file_exists(full_path, required=False)

    return all_required_exist

def check_imports():
    """检查导入是否正常"""
    print("\n" + "=" * 80)
    print("🔍 检查Python导入")
    print("=" * 80)

    import_tests = [
        ("agents.ppo", "PPOAgent"),
        ("agents.agents", "Agent"),
        ("networks.networks", "PPONetwork"),
        ("parameters.parameters", "DynamicParameter"),
        ("utils", None),
    ]

    all_imports_ok = True

    for module_path, class_name in import_tests:
        try:
            module = __import__(module_path, fromlist=[class_name] if class_name else [])
            if class_name and not hasattr(module, class_name):
                print(f"❌ {module_path}.{class_name} - 类不存在")
                all_imports_ok = False
            else:
                print(f"✅ {module_path}" + (f".{class_name}" if class_name else ""))
        except Exception as e:
            print(f"❌ {module_path} - 导入失败: {e}")
            all_imports_ok = False

    return all_imports_ok

def check_dependencies():
    """检查依赖是否安装"""
    print("\n" + "=" * 80)
    print("📦 检查Python依赖")
    print("=" * 80)

    dependencies = [
        "tensorflow",
        "tensorflow_probability",
        "numpy",
        "gym",
    ]

    optional_dependencies = [
        "cv2",
        "PIL",
        "matplotlib",
    ]

    print("\n必需依赖:")
    all_deps_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - 未安装")
            all_deps_ok = False

    print("\n可选依赖:")
    for dep in optional_dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"⚠️  {dep} - 未安装（可选）")

    return all_deps_ok

def count_lines_of_code():
    """统计代码行数"""
    print("\n" + "=" * 80)
    print("📊 代码统计")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    total_lines = 0
    file_count = 0

    for root, dirs, files in os.walk(base_dir):
        # 跳过__pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        file_count += 1
                except:
                    pass

    print(f"\n✅ Python文件总数: {file_count}")
    print(f"✅ 代码总行数: {total_lines:,}")

def main():
    """主函数"""
    print("=" * 80)
    print("🔍 RL Agent Only 完整性验证")
    print("=" * 80)

    # 检查目录结构
    structure_ok = check_directory_structure()

    # 检查导入
    imports_ok = check_imports()

    # 检查依赖
    deps_ok = check_dependencies()

    # 统计代码
    count_lines_of_code()

    # 总结
    print("\n" + "=" * 80)
    print("📋 验证总结")
    print("=" * 80)

    print(f"\n目录结构: {'✅ 通过' if structure_ok else '❌ 失败'}")
    print(f"Python导入: {'✅ 通过' if imports_ok else '❌ 失败'}")
    print(f"依赖检查: {'✅ 通过' if deps_ok else '❌ 失败'}")

    if structure_ok and imports_ok and deps_ok:
        print("\n🎉 所有检查通过！rl_agent_only已准备就绪。")
        print("\n📝 下一步:")
        print("   1. 查看 README.md 了解详细使用方法")
        print("   2. 运行 example_usage.py 测试基础功能")
        print("   3. 根据你的环境定制agent配置")
        return 0
    else:
        print("\n⚠️  部分检查未通过，请检查上述问题。")
        return 1

if __name__ == '__main__':
    sys.exit(main())
