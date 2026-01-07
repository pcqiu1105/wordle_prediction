"""
单元测试示例
"""

import unittest
import sys
import os

# 添加src路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import set_seed, ensure_dir


class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    def test_set_seed(self):
        """测试设置随机种子"""
        set_seed(42)
        # 这里可以添加更多具体的测试
        self.assertTrue(True)
    
    def test_ensure_dir(self):
        """测试目录创建"""
        test_dir = "test_directory"
        ensure_dir(test_dir)
        self.assertTrue(os.path.exists(test_dir))
        
        # 清理
        os.rmdir(test_dir)


if __name__ == '__main__':
    unittest.main()
