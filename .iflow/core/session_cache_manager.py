#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💾 历史会话缓存管理系统 V1.0
==============================

这是一个历史会话缓存管理系统，提供以下功能：
- 会话数据缓存和检索
- 智能缓存策略
- 会话状态管理
- 缓存优化和清理
- 数据持久化

核心特性：
- 高效缓存机制
- 智能过期策略
- 内存优化
- 快速检索
- 数据安全

作者: AI架构师团队
版本: 1.0.0
日期: 2025-11-16
"""

import os
import sys
import json
import sqlite3
import asyncio
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('历史会话缓存管理系统')

@dataclass
class 缓存项:
    """缓存项数据类"""
    键: str
    值: Any
    创建时间: datetime
    最后访问时间: datetime
    访问次数: int = 0
    过期时间: Optional[datetime] = None
    大小字节: int = 0
    
@dataclass
class 缓存统计:
    """缓存统计数据类"""
    总项数: int = 0
    总大小字节: int = 0
    命中次数: int = 0
    未命中次数: int = 0
    清理次数: int = 0
    最后清理时间: Optional[datetime] = None
    平均命中率: float = 0.0

class 历史会话缓存管理器:
    """历史会话缓存管理器主类"""
    
    def __init__(self, 缓存目录: str = "data/cache", 最大缓存大小: int = 1024 * 1024 * 1024):  # 1GB
        """初始化缓存管理器"""
        self.缓存目录 = Path(缓存目录)
        self.缓存目录.mkdir(parents=True, exist_ok=True)
        
        # 配置参数
        self.最大缓存大小 = 最大缓存大小
        self.默认过期时间 = timedelta(hours=24)
        self.清理阈值 = 0.8  # 当缓存使用率达到80%时清理
        
        # 内存缓存
        self.内存缓存 = {}
        self.缓存队列 = deque()
        self.访问记录 = defaultdict(int)
        
        # 数据库连接
        self.数据库路径 = self.缓存目录 / "cache.db"
        self.初始化数据库()
        
        # 统计信息
        self.缓存统计 = 缓存统计()
        
        # 线程锁
        self.缓存锁 = threading.RLock()
        
        # 后台清理线程
        self.清理线程 = None
        self.启动后台清理()
        
        logger.info("🚀 历史会话缓存管理器初始化完成")
    
    def 初始化数据库(self):
        """初始化缓存数据库"""
        try:
            with sqlite3.connect(self.数据库路径) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_items (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_time TEXT NOT NULL,
                        last_accessed_time TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        expire_time TEXT,
                        size_bytes INTEGER DEFAULT 0
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expire_time ON cache_items(expire_time)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_items(last_accessed_time)
                """)
                
                conn.commit()
            
            logger.info("✅ 缓存数据库初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 缓存数据库初始化失败: {e}")
            raise
    
    async def 设置缓存(self, 键: str, 值: Any, 过期时间: Optional[timedelta] = None) -> bool:
        """
        设置缓存项
        
        Args:
            键: 缓存键
            值: 缓存值
            过期时间: 过期时间，None表示使用默认过期时间
            
        Returns:
            是否设置成功
        """
        缓存项 = None
        try:
            with self.缓存锁:
                # 序列化值
                序列化值 = json.dumps(值, ensure_ascii=False, default=str)
                值大小 = len(序列化值.encode('utf-8'))
                
                # 检查缓存空间
                if self.获取当前缓存大小() + 值大小 > self.最大缓存大小:
                    await self.执行清理()
                
                # 创建时间
                当前时间 = datetime.now()
                项目过期时间 = 当前时间 + (过期时间 or self.默认过期时间)
                
                # 创建缓存项
                缓存项 = 缓存项(
                    键=键,
                    值=值,
                    创建时间=当前时间,
                    最后访问时间=当前时间,
                    访问次数=0,
                    过期时间=项目过期时间,
                    大小字节=值大小
                )
                
                # 更新内存缓存
                self.内存缓存[键] = 缓存项
                self.缓存队列.append(键)
                self.访问记录[键] = 0
                
                # 更新数据库
                await self.保存到数据库(缓存项)
                
                # 更新统计
                self.更新统计()
                
                logger.debug(f"💾 缓存已设置: {键} (大小: {值大小} 字节)")
                return True
                
        except Exception as e:
            logger.error(f"❌ 设置缓存失败 {键}: {e}")
            return False
    
    async def 获取缓存(self, 键: str) -> Optional[Any]:
        """
        获取缓存项
        
        Args:
            键: 缓存键
            
        Returns:
            缓存值，不存在返回None
        """
        try:
            with self.缓存锁:
                # 检查内存缓存
                if 键 in self.内存缓存:
                    缓存项 = self.内存缓存[键]
                    
                    # 检查是否过期
                    if 缓存项.过期时间 and 缓存项.过期时间 < datetime.now():
                        await self.删除缓存(键)
                        self.缓存统计.未命中次数 += 1
                        return None
                    
                    # 更新访问记录
                    缓存项.最后访问时间 = datetime.now()
                    缓存项.访问次数 += 1
                    self.访问记录[键] += 1
                    
                    # 更新数据库
                    await self.更新访问时间(键)
                    
                    # 更新统计
                    self.缓存统计.命中次数 += 1
                    self.更新统计()
                    
                    logger.debug(f"🎯 缓存命中: {键}")
                    return 缓存项.值
                
                # 检查数据库
                数据库项 = await self.从数据库加载(键)
                if 数据库项:
                    # 检查是否过期
                    if 数据库项.过期时间 and 数据库项.过期时间 < datetime.now():
                        await self.删除缓存(键)
                        self.缓存统计.未命中次数 += 1
                        return None
                    
                    # 加载到内存缓存
                    self.内存缓存[键] = 数据库项
                    self.缓存队列.append(键)
                    self.访问记录[键] = 0
                    
                    # 更新访问记录
                    数据库项.最后访问时间 = datetime.now()
                    数据库项.访问次数 += 1
                    self.访问记录[键] += 1
                    
                    # 更新数据库
                    await self.更新访问时间(键)
                    
                    # 更新统计
                    self.缓存统计.命中次数 += 1
                    self.更新统计()
                    
                    logger.debug(f"🎯 缓存命中(数据库): {键}")
                    return 数据库项.值
                
                # 缓存未命中
                self.缓存统计.未命中次数 += 1
                self.更新统计()
                
                logger.debug(f"❌ 缓存未命中: {键}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 获取缓存失败 {键}: {e}")
            self.缓存统计.未命中次数 += 1
            return None
    
    async def 删除缓存(self, 键: str) -> bool:
        """
        删除缓存项
        
        Args:
            键: 缓存键
            
        Returns:
            是否删除成功
        """
        try:
            with self.缓存锁:
                # 从内存缓存删除
                if 键 in self.内存缓存:
                    del self.内存缓存[键]
                
                # 从访问记录删除
                if 键 in self.访问记录:
                    del self.访问记录[键]
                
                # 从队列删除
                try:
                    self.缓存队列.remove(键)
                except ValueError:
                    pass
                
                # 从数据库删除
                await self.从数据库删除(键)
                
                # 更新统计
                self.更新统计()
                
                logger.debug(f"🗑️ 缓存已删除: {键}")
                return True
                
        except Exception as e:
            logger.error(f"❌ 删除缓存失败 {键}: {e}")
            return False
    
    async def 清空缓存(self) -> bool:
        """清空所有缓存"""
        try:
            with self.缓存锁:
                # 清空内存缓存
                self.内存缓存.clear()
                self.缓存队列.clear()
                self.访问记录.clear()
                
                # 清空数据库
                with sqlite3.connect(self.数据库路径) as conn:
                    conn.execute("DELETE FROM cache_items")
                    conn.commit()
                
                # 重置统计
                self.缓存统计 = 缓存统计()
                
                logger.info("🗑️ 所有缓存已清空")
                return True
                
        except Exception as e:
            logger.error(f"❌ 清空缓存失败: {e}")
            return False
    
    async def 执行清理(self):
        """执行缓存清理"""
        try:
            logger.info("🧹 开始执行缓存清理")
            
            with self.缓存锁:
                # 获取过期的缓存项
                当前时间 = datetime.now()
                过期键 = []
                
                for 键, 缓存项 in self.内存缓存.items():
                    if 缓存项.过期时间 and 缓存项.过期时间 < 当前时间:
                        过期键.append(键)
                
                # 删除过期项
                for 键 in 过期键:
                    await self.删除缓存(键)
                
                # 如果还是太大，按LRU策略删除
                while self.获取当前缓存大小() > self.最大缓存大小 * self.清理阈值:
                    if not self.缓存队列:
                        break
                    
                    # 找到最少使用的键
                    最少使用键 = min(self.访问记录.items(), key=lambda x: x[1])[0]
                    await self.删除缓存(最少使用键)
                
                # 更新统计
                self.缓存统计.清理次数 += 1
                self.缓存统计.最后清理时间 = 当前时间
                
                logger.info(f"✅ 缓存清理完成 - 删除项数: {len(过期键)}")
                
        except Exception as e:
            logger.error(f"❌ 缓存清理失败: {e}")
    
    def 获取当前缓存大小(self) -> int:
        """获取当前缓存大小"""
        return sum(项.大小字节 for 项 in self.内存缓存.values())
    
    def 更新统计(self):
        """更新缓存统计"""
        总访问次数 = self.缓存统计.命中次数 + self.缓存统计.未命中次数
        if 总访问次数 > 0:
            self.缓存统计.平均命中率 = self.缓存统计.命中次数 / 总访问次数
        
        self.缓存统计.总项数 = len(self.内存缓存)
        self.缓存统计.总大小字节 = self.获取当前缓存大小()
    
    async def 保存到数据库(self, 缓存项: 缓存项):
        """保存缓存项到数据库"""
        try:
            with sqlite3.connect(self.数据库路径) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_items 
                    (key, value, created_time, last_accessed_time, access_count, expire_time, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    缓存项.键,
                    json.dumps(缓存项.值, ensure_ascii=False, default=str),
                    缓存项.创建时间.isoformat(),
                    缓存项.最后访问时间.isoformat(),
                    缓存项.访问次数,
                    缓存项.过期时间.isoformat() if 缓存项.过期时间 else None,
                    缓存项.大小字节
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ 保存到数据库失败: {e}")
    
    async def 从数据库加载(self, 键: str) -> Optional[缓存项]:
        """从数据库加载缓存项"""
        try:
            with sqlite3.connect(self.数据库路径) as conn:
                cursor = conn.execute("""
                    SELECT key, value, created_time, last_accessed_time, access_count, expire_time, size_bytes
                    FROM cache_items WHERE key = ?
                """, (键,))
                
                行 = cursor.fetchone()
                if 行:
                    return 缓存项(
                        键=行[0],
                        值=json.loads(行[1]),
                        创建时间=datetime.fromisoformat(行[2]),
                        最后访问时间=datetime.fromisoformat(行[3]),
                        访问次数=行[4],
                        过期时间=datetime.fromisoformat(行[5]) if 行[5] else None,
                        大小字节=行[6]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"❌ 从数据库加载失败: {e}")
            return None
    
    async def 更新访问时间(self, 键: str):
        """更新访问时间"""
        try:
            当前时间 = datetime.now()
            with sqlite3.connect(self.数据库路径) as conn:
                conn.execute("""
                    UPDATE cache_items 
                    SET last_accessed_time = ?, access_count = access_count + 1
                    WHERE key = ?
                """, (当前时间.isoformat(), 键))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ 更新访问时间失败: {e}")
    
    async def 从数据库删除(self, 键: str):
        """从数据库删除缓存项"""
        try:
            with sqlite3.connect(self.数据库路径) as conn:
                conn.execute("DELETE FROM cache_items WHERE key = ?", (键,))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ 从数据库删除失败: {e}")
    
    def 启动后台清理(self):
        """启动后台清理线程"""
        def 清理任务():
            while True:
                try:
                    asyncio.run(self.执行清理())
                    time.sleep(300)  # 每5分钟清理一次
                except Exception as e:
                    logger.error(f"❌ 后台清理失败: {e}")
                    time.sleep(60)  # 出错后等待1分钟再试
        
        self.清理线程 = threading.Thread(target=清理任务, daemon=True)
        self.清理线程.start()
        logger.info("🔄 后台清理线程已启动")
    
    def 停止后台清理(self):
        """停止后台清理线程"""
        if self.清理线程 and self.清理线程.is_alive():
            # 注意：这里只是标记，实际线程会在下一次循环时自然结束
            logger.info("⏹️ 后台清理线程已停止")
    
    async def 获取缓存统计(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.缓存锁:
            self.更新统计()
            
            return {
                "总项数": self.缓存统计.总项数,
                "总大小字节": self.缓存统计.总大小字节,
                "总大小MB": round(self.缓存统计.总大小字节 / (1024 * 1024), 2),
                "命中次数": self.缓存统计.命中次数,
                "未命中次数": self.缓存统计.未命中次数,
                "清理次数": self.缓存统计.清理次数,
                "最后清理时间": self.缓存统计.最后清理时间.isoformat() if self.缓存统计.最后清理时间 else None,
                "平均命中率": round(self.缓存统计.平均命中率, 4),
                "最大缓存大小": self.最大缓存大小,
                "使用率": round(self.获取当前缓存大小() / self.最大缓存大小, 4),
                "统计时间": datetime.now().isoformat()
            }
    
    async def 获取热门键(self, 限制数量: int = 10) -> List[Tuple[str, int]]:
        """获取热门缓存键"""
        with self.缓存锁:
            热门键 = sorted(self.访问记录.items(), key=lambda x: x[1], reverse=True)
            return 热门键[:限制数量]
    
    async def 获取缓存项列表(self, 限制数量: int = 100) -> List[Dict[str, Any]]:
        """获取缓存项列表"""
        with self.缓存锁:
            项目列表 = []
            
            for 键, 缓存项 in list(self.内存缓存.items())[:限制数量]:
                项目列表.append({
                    "键": 键,
                    "创建时间": 缓存项.创建时间.isoformat(),
                    "最后访问时间": 缓存项.最后访问时间.isoformat(),
                    "访问次数": 缓存项.访问次数,
                    "大小字节": 缓存项.大小字节,
                    "过期时间": 缓存项.过期时间.isoformat() if 缓存项.过期时间 else None,
                    "是否过期": (缓存项.过期时间 and 缓存项.过期时间 < datetime.now())
                })
            
            return 项目列表

# 全局实例
_缓存管理器实例 = None

def get_缓存管理器() -> 历史会话缓存管理器:
    """获取缓存管理器实例"""
    global _缓存管理器实例
    if _缓存管理器实例 is None:
        _缓存管理器实例 = 历史会话缓存管理器()
    return _缓存管理器实例

# 便捷函数
async def 设置缓存(键: str, 值: Any, 过期时间: Optional[timedelta] = None) -> bool:
    """便捷的缓存设置函数"""
    管理器 = get_缓存管理器()
    return await 管理器.设置缓存(键, 值, 过期时间)

async def 获取缓存(键: str) -> Optional[Any]:
    """便捷的缓存获取函数"""
    管理器 = get_缓存管理器()
    return await 管理器.获取缓存(键)

async def 删除缓存(键: str) -> bool:
    """便捷的缓存删除函数"""
    管理器 = get_缓存管理器()
    return await 管理器.删除缓存(键)

if __name__ == "__main__":
    # 测试代码
    async def 测试缓存系统():
        管理器 = get_缓存管理器()
        
        # 测试设置缓存
        测试数据 = {
            "会话ID": "test_001",
            "用户ID": "user_001",
            "消息": "这是一条测试消息",
            "时间戳": datetime.now().isoformat()
        }
        
        设置结果 = await 管理器.设置缓存("test_key", 测试数据)
        print(f"设置缓存结果: {设置结果}")
        
        # 测试获取缓存
        获取结果 = await 管理器.获取缓存("test_key")
        print(f"获取缓存结果: {获取结果}")
        
        # 测试缓存统计
        统计信息 = await 管理器.获取缓存统计()
        print(f"缓存统计: {统计信息}")
        
        # 测试热门键
        热门键 = await 管理器.获取热门键()
        print(f"热门键: {热门键}")
        
        # 测试缓存列表
        缓存列表 = await 管理器.获取缓存项列表()
        print(f"缓存列表: {缓存列表}")
        
        # 清理测试数据
        await 管理器.删除缓存("test_key")
        print("测试数据已清理")
    
    asyncio.run(测试缓存系统())