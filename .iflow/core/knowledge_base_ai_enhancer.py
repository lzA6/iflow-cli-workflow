
# 魔法数字常量定义
HTTP_INTERNAL_ERROR = 500
MAGIC_NUMBER_4000 = 4000
MAGIC_NUMBER_3000 = 3000
MAGIC_NUMBER_8192 = 8192
MAGIC_NUMBER_100000 = 100000


# 魔法数字常量定义
HTTP_INTERNAL_ERROR = 500
MAGIC_NUMBER_4000 = 4000
MAGIC_NUMBER_3000 = 3000
MAGIC_NUMBER_8192 = 8192
MAGIC_NUMBER_100000 = 100000


# 魔法数字常量定义
HTTP_INTERNAL_ERROR = HTTP_INTERNAL_ERROR
MAGIC_NUMBER_4000 = MAGIC_NUMBER_4000
MAGIC_NUMBER_3000 = MAGIC_NUMBER_3000
MAGIC_NUMBER_8192 = MAGIC_NUMBER_8192
MAGIC_NUMBER_100000 = MAGIC_NUMBER_100000


# 魔法数字常量定义
HTTP_INTERNAL_ERROR = 500
MAGIC_NUMBER_4000 = 4000
MAGIC_NUMBER_3000 = 3000
MAGIC_NUMBER_8192 = 8192
MAGIC_NUMBER_100000 = 100000

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库AI增强器
提供AI总结、汉化、优化等功能
"""

import os
import json
import asyncio
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import zipfile
import tarfile
import gzip
import shutil

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_ROOT = PROJECT_ROOT / "knowledge_base"

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class AIEnhancementConfig:
    """AI增强配置"""
    summary_model: str = "gpt-3.5-turbo"
    translation_model: str = "gpt-3.5-turbo"
    optimization_model: str = "gpt-4"
    max_summary_length: int = 500
    batch_size: int = 10
    api_base: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    enable_local_llm: bool = True
    local_llm_path: str = "./models"

class KnowledgeBaseAIEnhancer:
    """知识库AI增强器"""
    
    def __init__(self, config: Optional[AIEnhancementConfig] = None):
        self.config = config or AIEnhancementConfig()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 统计信息
        self.stats = {
            "summaries_generated": 0,
            "translations_completed": 0,
            "optimizations_applied": 0,
            "repositories_imported": 0,
            "total_processing_time": 0.0
        }
        
        # 支持的文件类型
        self.supported_archive_types = {
            '.zip': self._extract_zip,
            '.tar': self._extract_tar,
            '.tar.gz': self._extract_tar_gz,
            '.tgz': self._extract_tar_gz,
            '.gz': self._extract_gz
        }
        
        # 代码文件类型
        self.code_file_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
            '.html', '.css', '.scss', '.less', '.vue', '.jsx', '.tsx',
            '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.sql', '.sh', '.bat', '.ps1', '.dockerfile', 'dockerfile'
        }
        
        # 文档文件类型
        self.doc_file_extensions = {
            '.md', '.txt', '.rst', '.adoc', '.tex', '.pdf', '.doc', '.docx'
        }
    
    async def generate_summary(self, content: str, title: str = "", language: str = "zh") -> Dict[str, Any]:
        """生成内容总结"""
        start_time = time.time()
        
        try:
            # 构建总结提示
            if language == "zh":
                prompt = f"""请为以下内容生成一个简洁准确的中文总结（{self.config.max_summary_length}字以内）：

标题：{title}

内容：
{content[:4000]}

总结要求：
1. 准确概括主要内容
2. 突出关键信息
3. 保持逻辑清晰
4. 使用简洁的中文表达

请直接返回总结内容，不要包含其他解释。"""
            else:
                prompt = f"""Please generate a concise summary for the following content (within {self.config.max_summary_length} words):

Title: {title}

Content:
{content[:4000]}

Requirements:
1. Accurately summarize the main content
2. Highlight key information
3. Maintain logical clarity
4. Use concise expression

Please return only the summary content without additional explanations."""
            
            # 调用AI生成总结
            summary = await self._call_ai_api(prompt)
            
            # 更新统计
            self.stats["summaries_generated"] += 1
            self.stats["total_processing_time"] += time.time() - start_time
            
            return {
                "success": True,
                "summary": summary,
                "original_length": len(content),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(content) if content else 0,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"生成总结失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def translate_to_chinese(self, content: str, source_lang: str = "auto") -> Dict[str, Any]:
        """翻译内容到中文"""
        start_time = time.time()
        
        try:
            # 构建翻译提示
            prompt = f"""请将以下内容准确翻译成中文，保持原文的格式和结构：

原文语言：{source_lang}
原文内容：
{content[:4000]}

翻译要求：
1. 准确传达原文含义
2. 保持专业术语的一致性
3. 适应中文表达习惯
4. 保留原有的格式和结构

请直接返回翻译结果，不要包含其他解释。"""
            
            # 调用AI翻译
            translation = await self._call_ai_api(prompt)
            
            # 更新统计
            self.stats["translations_completed"] += 1
            self.stats["total_processing_time"] += time.time() - start_time
            
            return {
                "success": True,
                "translation": translation,
                "source_lang": source_lang,
                "target_lang": "zh",
                "original_length": len(content),
                "translation_length": len(translation),
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"翻译失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def optimize_content(self, content: str, content_type: str = "general") -> Dict[str, Any]:
        """优化内容质量和结构"""
        start_time = time.time()
        
        try:
            # 根据内容类型构建优化提示
            if content_type == "code":
                prompt = f"""请优化以下代码的质量和结构：

代码内容：
{content[:3000]}

优化要求：
1. 添加必要的注释说明
2. 优化代码结构和逻辑
3. 确保代码可读性
4. 保持原有功能不变
5. 如果是中文注释，请保持中文

请直接返回优化后的代码，不要包含其他解释。"""
            elif content_type == "documentation":
                prompt = f"""请优化以下文档的结构和表达：

文档内容：
{content[:4000]}

优化要求：
1. 改善文档结构和层次
2. 优化语言表达
3. 确保逻辑清晰
4. 添加必要的标题和标记
5. 保持中文表达习惯

请直接返回优化后的文档，不要包含其他解释。"""
            else:
                prompt = f"""请优化以下内容的质量和表达：

内容：
{content[:4000]}

优化要求：
1. 改善语言表达
2. 优化逻辑结构
3. 确保内容准确
4. 提高可读性
5. 保持原意不变

请直接返回优化后的内容，不要包含其他解释。"""
            
            # 调用AI优化
            optimized = await self._call_ai_api(prompt)
            
            # 更新统计
            self.stats["optimizations_applied"] += 1
            self.stats["total_processing_time"] += time.time() - start_time
            
            return {
                "success": True,
                "optimized_content": optimized,
                "content_type": content_type,
                "original_length": len(content),
                "optimized_length": len(optimized),
                "improvement_score": self._calculate_improvement_score(content, optimized),
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"内容优化失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def import_github_repository(self, repo_url: str, target_group: str = "github_imports") -> Dict[str, Any]:
        """导入GitHub仓库到知识库"""
        start_time = time.time()
        
        try:
            # 解析仓库URL
            repo_info = self._parse_github_url(repo_url)
            if not repo_info:
                return {
                    "success": False,
                    "error": "无效的GitHub仓库URL"
                }
            
            # 下载仓库
            repo_path = await self._download_github_repo(repo_info)
            
            # 解压仓库
            extracted_path = await self._extract_repository(repo_path)
            
            # 处理仓库文件
            import_result = await self._process_repository_files(extracted_path, target_group)
            
            # 清理临时文件
            shutil.rmtree(repo_path.parent, ignore_errors=True)
            
            # 更新统计
            self.stats["repositories_imported"] += 1
            self.stats["total_processing_time"] += time.time() - start_time
            
            return {
                "success": True,
                "repo_info": repo_info,
                "import_result": import_result,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"导入GitHub仓库失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def batch_process_documents(self, documents: List[Dict[str, Any]], operations: List[str]) -> Dict[str, Any]:
        """批量处理文档"""
        start_time = time.time()
        results = []
        
        for doc in documents:
            doc_result = {"doc_id": doc.get("id", ""), "operations": {}}
            
            # 执行每个操作
            for operation in operations:
                if operation == "summarize":
                    result = await self.generate_summary(
                        doc.get("content", ""),
                        doc.get("title", ""),
                        doc.get("language", "zh")
                    )
                    doc_result["operations"]["summarize"] = result
                
                elif operation == "translate":
                    result = await self.translate_to_chinese(
                        doc.get("content", ""),
                        doc.get("source_lang", "auto")
                    )
                    doc_result["operations"]["translate"] = result
                
                elif operation == "optimize":
                    result = await self.optimize_content(
                        doc.get("content", ""),
                        doc.get("content_type", "general")
                    )
                    doc_result["operations"]["optimize"] = result
            
            results.append(doc_result)
        
        return {
            "success": True,
            "processed_count": len(results),
            "results": results,
            "total_time": time.time() - start_time
        }
    
    async def _call_ai_api(self, prompt: str) -> str:
        """调用AI API"""
        # 这里可以集成OpenAI API或其他AI服务
        # 为了演示，返回模拟结果
        await asyncio.sleep(0.5)  # 模拟API调用延迟
        
        # 简单的模拟响应
        if "总结" in prompt or "summary" in prompt:
            return f"这是对内容的总结：{prompt[:100]}..."
        elif "翻译" in prompt or "translate" in prompt:
            return f"这是翻译结果：{prompt[:100]}..."
        elif "优化" in prompt or "optimize" in prompt:
            return f"这是优化后的内容：{prompt[:100]}..."
        else:
            return "AI处理完成"
    
    def _calculate_improvement_score(self, original: str, optimized: str) -> float:
        """计算改进分数"""
        # 简单的改进分数计算
        length_ratio = len(optimized) / len(original) if original else 1
        # 这里可以添加更复杂的评分逻辑
        return min(1.0, max(0.0, 1.0 - abs(1.0 - length_ratio)))
    
    def _parse_github_url(self, url: str) -> Optional[Dict[str, str]]:
        """解析GitHub URL"""
        # 支持多种GitHub URL格式
        import re
        
        patterns = [
            r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$",
        ]
        
        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                return {
                    "owner": match.group(1),
                    "repo": match.group(2),
                    "url": url
                }
        
        return None
    
    async def _download_github_repo(self, repo_info: Dict[str, str]) -> Path:
        """下载GitHub仓库"""
        temp_dir = KNOWLEDGE_BASE_ROOT / "temp" / f"github_{repo_info['owner']}_{repo_info['repo']}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建下载URL
        download_url = f"https://github.com/{repo_info['owner']}/{repo_info['repo']}/archive/main.zip"
        
        # 下载文件
        zip_path = temp_dir / "repo.zip"
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return zip_path
    
    async def _extract_repository(self, repo_path: Path) -> Path:
        """解压仓库"""
        extract_dir = repo_path.parent / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(repo_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # 返回实际的项目目录（去掉版本号前缀）
        items = list(extract_dir.iterdir())
        if items and items[0].is_dir():
            return items[0]
        
        return extract_dir
    
    async def _process_repository_files(self, repo_path: Path, target_group: str) -> Dict[str, Any]:
        """处理仓库文件"""
        processed_files = []
        total_size = 0
        
        # 遍历仓库文件
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                # 跳过某些文件类型
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    # 读取文件内容
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # 限制文件大小
                    if len(content) > 100000:  # 100KB
                        content = content[:100000]
                    
                    # 确定文件类型
                    file_ext = file_path.suffix.lower()
                    content_type = "code" if file_ext in self.code_file_extensions else "document"
                    
                    processed_files.append({
                        "path": str(file_path.relative_to(repo_path)),
                        "content": content,
                        "size": len(content),
                        "type": content_type,
                        "extension": file_ext
                    })
                    
                    total_size += len(content)
                    
                except Exception as e:
                    logger.warning(f"处理文件失败 {file_path}: {e}")
                    continue
        
        return {
            "processed_files": len(processed_files),
            "total_size": total_size,
            "file_types": self._analyze_file_types(processed_files),
            "files": processed_files[:10]  # 只返回前10个文件的详细信息
        }
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        # 跳过的目录
        skip_dirs = {'.git', '.svn', '__pycache__', 'node_modules', '.vscode', '.idea', 'dist', 'build'}
        
        # 检查是否在跳过目录中
        for part in file_path.parts:
            if part in skip_dirs:
                return True
        
        # 跳过的文件扩展名
        skip_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', '.img', '.iso', '.zip', '.tar', '.gz'}
        
        if file_path.suffix.lower() in skip_extensions:
            return True
        
        # 跳过隐藏文件
        if file_path.name.startswith('.'):
            return True
        
        return False
    
    def _analyze_file_types(self, files: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析文件类型分布"""
        type_counts = {}
        
        for file_info in files:
            file_type = file_info.get("type", "unknown")
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        return type_counts
    
    def _extract_zip(self, archive_path: Path, extract_to: Path):
        """解压ZIP文件"""
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    def _extract_tar(self, archive_path: Path, extract_to: Path):
        """解压TAR文件"""
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    
    def _extract_tar_gz(self, archive_path: Path, extract_to: Path):
        """解压TAR.GZ文件"""
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    
    def _extract_gz(self, archive_path: Path, extract_to: Path):
        """解压GZ文件"""
        output_path = extract_to / archive_path.stem
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    async def extract_archive(self, archive_path: str, extract_to: str) -> Dict[str, Any]:
        """解压归档文件"""
        start_time = time.time()
        
        try:
            archive_path = Path(archive_path)
            extract_to = Path(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            
            file_ext = archive_path.suffix.lower()
            
            if file_ext not in self.supported_archive_types:
                return {
                    "success": False,
                    "error": f"不支持的文件类型: {file_ext}"
                }
            
            # 执行解压
            self.supported_archive_types[file_ext](archive_path, extract_to)
            
            # 统计解压结果
            extracted_files = list(extract_to.rglob("*"))
            total_size = sum(f.stat().st_size for f in extracted_files if f.is_file())
            
            return {
                "success": True,
                "extracted_files": len([f for f in extracted_files if f.is_file()]),
                "total_size": total_size,
                "extracted_to": str(extract_to),
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"解压失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enhancement_stats": self.stats,
            "memory_usage": self._get_memory_usage(),
            "supported_formats": {
                "archives": list(self.supported_archive_types.keys()),
                "code_files": list(self.code_file_extensions),
                "doc_files": list(self.doc_file_extensions)
            }
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }

# 全局实例
_ai_enhancer = None

def get_ai_enhancer() -> KnowledgeBaseAIEnhancer:
    """获取AI增强器实例"""
    global _ai_enhancer
    if _ai_enhancer is None:
        _ai_enhancer = KnowledgeBaseAIEnhancer()
    return _ai_enhancer