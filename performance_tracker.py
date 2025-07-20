"""
Performance Tracker for MoE System
Tracks model response times and performance metrics with intelligent caching (10 logs max)
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import logging

@dataclass
class PerformanceLog:
    timestamp: float
    model_name: str
    task_type: str
    strategy: str
    response_time: float
    success: bool
    content_length: int
    complexity_level: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PerformanceTracker:
    def __init__(self, max_logs: int = 10):
        self.max_logs = max_logs
        self.logs = deque(maxlen=max_logs)  # Automatically maintains max size
        self.logger = logging.getLogger(__name__)
        
        # Performance statistics cache
        self._stats_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
        
    def log_performance(self, model_name: str, task_type: str, strategy: str,
                       response_time: float, success: bool, content_length: int = 0,
                       complexity_level: str = "moderate", parameters: Dict[str, Any] = None):
        """Log a performance entry"""
        
        log_entry = PerformanceLog(
            timestamp=time.time(),
            model_name=model_name,
            task_type=task_type,
            strategy=strategy,
            response_time=response_time,
            success=success,
            content_length=content_length,
            complexity_level=complexity_level,
            parameters=parameters or {}
        )
        
        self.logs.append(log_entry)
        
        # Log important metrics
        if response_time > 30:  # Slow response
            self.logger.warning(f"⚠️ Slow response: {model_name} took {response_time:.2f}s")
        elif response_time < 2:  # Fast response
            self.logger.info(f"🚀 Fast response: {model_name} took {response_time:.2f}s")
        
        # Clear stats cache to force recalculation
        self._stats_cache.clear()
        
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific model"""
        
        model_logs = [log for log in self.logs if log.model_name == model_name]
        
        if not model_logs:
            return {"error": "No performance data available"}
        
        response_times = [log.response_time for log in model_logs if log.success]
        success_rate = sum(1 for log in model_logs if log.success) / len(model_logs)
        
        return {
            "model": model_name,
            "total_requests": len(model_logs),
            "success_rate": success_rate,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "recent_performance": response_times[-3:] if len(response_times) >= 3 else response_times
        }
    
    def get_fastest_models_for_task(self, task_type: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get fastest performing models for a specific task type"""
        
        task_logs = [log for log in self.logs if log.task_type == task_type and log.success]
        
        if not task_logs:
            return []
        
        # Group by model and calculate averages
        model_performance = {}
        for log in task_logs:
            if log.model_name not in model_performance:
                model_performance[log.model_name] = []
            model_performance[log.model_name].append(log.response_time)
        
        # Calculate averages and sort by speed
        model_averages = []
        for model, times in model_performance.items():
            avg_time = sum(times) / len(times)
            model_averages.append({
                "model": model,
                "avg_response_time": avg_time,
                "sample_count": len(times)
            })
        
        return sorted(model_averages, key=lambda x: x["avg_response_time"])[:limit]
    
    def should_use_model(self, model_name: str, urgency: str = "normal") -> bool:
        """Intelligent decision on whether to use a model based on performance"""
        
        perf = self.get_model_performance(model_name)
        
        if "error" in perf:
            return True  # No data, allow usage
        
        # Decision logic based on urgency and performance
        if urgency == "fast" and perf["avg_response_time"] > 10:
            return False  # Too slow for fast requests
        elif urgency == "quality" and perf["success_rate"] < 0.8:
            return False  # Too unreliable for quality requests
        
        return True
    
    def get_intelligent_recommendations(self, task_type: str) -> Dict[str, Any]:
        """Get intelligent model recommendations based on performance data"""
        
        current_time = time.time()
        cache_key = f"recommendations_{task_type}"
        
        # Use cached results if available and fresh
        if (cache_key in self._stats_cache and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._stats_cache[cache_key]
        
        fastest_models = self.get_fastest_models_for_task(task_type)
        
        recommendations = {
            "task_type": task_type,
            "fastest_models": fastest_models,
            "timestamp": current_time,
            "data_points": len([log for log in self.logs if log.task_type == task_type])
        }
        
        # Add speed recommendations
        if fastest_models:
            fastest = fastest_models[0]
            if fastest["avg_response_time"] < 5:
                recommendations["speed_tier"] = "fast"
                recommendations["recommendation"] = f"Use {fastest['model']} for speed"
            elif fastest["avg_response_time"] < 15:
                recommendations["speed_tier"] = "moderate" 
                recommendations["recommendation"] = f"Use {fastest['model']} for balanced performance"
            else:
                recommendations["speed_tier"] = "slow"
                recommendations["recommendation"] = "Consider using multiple models for this task"
        
        # Cache the results
        self._stats_cache[cache_key] = recommendations
        self._cache_timestamp = current_time
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        
        if not self.logs:
            return {"error": "No performance data available"}
        
        total_requests = len(self.logs)
        successful_requests = sum(1 for log in self.logs if log.success)
        
        # Group by model
        model_stats = {}
        for log in self.logs:
            if log.model_name not in model_stats:
                model_stats[log.model_name] = {"times": [], "successes": 0, "total": 0}
            
            model_stats[log.model_name]["total"] += 1
            if log.success:
                model_stats[log.model_name]["times"].append(log.response_time)
                model_stats[log.model_name]["successes"] += 1
        
        # Calculate model averages
        for model, stats in model_stats.items():
            if stats["times"]:
                stats["avg_time"] = sum(stats["times"]) / len(stats["times"])
                stats["success_rate"] = stats["successes"] / stats["total"]
            else:
                stats["avg_time"] = 0
                stats["success_rate"] = 0
        
        return {
            "total_requests": total_requests,
            "success_rate": successful_requests / total_requests,
            "models_tested": len(model_stats),
            "model_performance": model_stats,
            "latest_logs": len(self.logs),
            "max_capacity": self.max_logs
        }
    
    def export_logs(self) -> List[Dict[str, Any]]:
        """Export logs as JSON-serializable format"""
        return [log.to_dict() for log in self.logs]
    
    def clear_logs(self):
        """Clear all performance logs"""
        self.logs.clear()
        self._stats_cache.clear()
        self.logger.info("🧹 Performance logs cleared")

# Global performance tracker instance
performance_tracker = PerformanceTracker(max_logs=10)
