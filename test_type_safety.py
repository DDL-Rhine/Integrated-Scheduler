#!/usr/bin/env python3
"""
测试类型安全性的简单脚本
"""

def test_basic_operations():
    """测试基本的列表和元组操作"""
    
    # 测试1: 列表切片
    donor_allocation = [1, 2, 3, 4]
    transfer_amount = 2
    
    try:
        transferred_nodes = donor_allocation[-transfer_amount:]
        remaining_nodes = donor_allocation[:-transfer_amount]
        print(f"✓ 列表切片成功: transferred={transferred_nodes}, remaining={remaining_nodes}")
    except Exception as e:
        print(f"✗ 列表切片失败: {e}")
    
    # 测试2: 列表连接
    help_allocation = [5, 6]
    transferred_nodes = [3, 4]
    
    try:
        new_allocation = help_allocation + transferred_nodes
        print(f"✓ 列表连接成功: {new_allocation}")
    except Exception as e:
        print(f"✗ 列表连接失败: {e}")
    
    # 测试3: 元组和列表混合（可能的问题源）
    tuple_allocation = (1, 2, 3, 4)
    list_transfer = [3, 4]
    
    try:
        # 这会导致错误
        result = tuple_allocation + list_transfer
        print(f"✓ 元组列表连接: {result}")
    except TypeError as e:
        print(f"✗ 元组列表连接失败 (预期): {e}")
        # 正确的方式
        result = list(tuple_allocation) + list_transfer
        print(f"✓ 修正后连接成功: {result}")

if __name__ == "__main__":
    test_basic_operations()
    print("类型安全测试完成")
