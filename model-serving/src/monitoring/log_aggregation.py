"""
Log aggregation and analysis utilities.

This module provides tools for collecting, analyzing, and aggregating logs
from the CarlaRL serving infrastructure for monitoring and debugging purposes.
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict, Counter
import argparse
import sys


@dataclass
class LogEntry:
    """Represents a structured log entry."""
    timestamp: datetime
    level: str
    message: str
    correlation_id: Optional[str]
    logger: str
    event_type: Optional[str]
    fields: Dict[str, Any]
    raw_line: str


class LogAggregator:
    """
    Log aggregator for analyzing CarlaRL serving logs.
    
    Provides capabilities for:
    - Parsing structured JSON logs
    - Filtering and searching logs
    - Performance analysis
    - Error analysis and reporting
    - Correlation tracking
    """
    
    def __init__(self):
        """Initialize log aggregator."""
        self.entries: List[LogEntry] = []
        self.correlation_map: Dict[str, List[LogEntry]] = defaultdict(list)
        self.event_counts: Counter = Counter()
        self.error_counts: Counter = Counter()
    
    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry."""
        try:
            # Try to parse as JSON first
            log_data = json.loads(line.strip())
            
            # Extract timestamp
            timestamp_str = log_data.get('timestamp', '')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now(timezone.utc)
            
            # Extract fields
            level = log_data.get('level', 'INFO')
            message = log_data.get('message', '')
            correlation_id = log_data.get('correlation_id')
            logger = log_data.get('logger', 'unknown')
            event_type = log_data.get('event_type')
            
            # Extract additional fields
            fields = {k: v for k, v in log_data.items() 
                     if k not in {'timestamp', 'level', 'message', 'correlation_id', 'logger', 'event_type'}}
            
            entry = LogEntry(
                timestamp=timestamp,
                level=level,
                message=message,
                correlation_id=correlation_id,
                logger=logger,
                event_type=event_type,
                fields=fields,
                raw_line=line
            )
            
            return entry
            
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback to simple text parsing
            return self._parse_text_log(line)
    
    def _parse_text_log(self, line: str) -> Optional[LogEntry]:
        """Parse a simple text log line."""
        # Simple regex for common log formats
        patterns = [
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*Z?)\s+(\w+)\s+(\w+)\s+(.+)',
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(\w+)\s+(\w+)\s+(.+)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                timestamp_str, level, logger, message = match.groups()
                
                # Parse timestamp
                try:
                    if 'T' in timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                except ValueError:
                    timestamp = datetime.now(timezone.utc)
                
                entry = LogEntry(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    correlation_id=None,
                    logger=logger,
                    event_type=None,
                    fields={},
                    raw_line=line
                )
                
                return entry
        
        return None
    
    def add_log_line(self, line: str):
        """Add a log line to the aggregator."""
        entry = self.parse_log_line(line)
        if entry:
            self.entries.append(entry)
            
            # Update indices
            if entry.correlation_id:
                self.correlation_map[entry.correlation_id].append(entry)
            
            if entry.event_type:
                self.event_counts[entry.event_type] += 1
            
            if entry.level in ['ERROR', 'CRITICAL']:
                self.error_counts[entry.message] += 1
    
    def load_from_file(self, file_path: str):
        """Load logs from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.add_log_line(line)
    
    def load_from_stdin(self):
        """Load logs from stdin."""
        for line in sys.stdin:
            self.add_log_line(line)
    
    def filter_by_time_range(self, start_time: datetime, end_time: datetime) -> List[LogEntry]:
        """Filter entries by time range."""
        return [entry for entry in self.entries 
                if start_time <= entry.timestamp <= end_time]
    
    def filter_by_level(self, level: str) -> List[LogEntry]:
        """Filter entries by log level."""
        return [entry for entry in self.entries if entry.level == level]
    
    def filter_by_event_type(self, event_type: str) -> List[LogEntry]:
        """Filter entries by event type."""
        return [entry for entry in self.entries if entry.event_type == event_type]
    
    def filter_by_correlation_id(self, correlation_id: str) -> List[LogEntry]:
        """Filter entries by correlation ID."""
        return self.correlation_map.get(correlation_id, [])
    
    def search_by_message(self, pattern: str) -> List[LogEntry]:
        """Search entries by message pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        return [entry for entry in self.entries 
                if regex.search(entry.message)]
    
    def get_correlation_trace(self, correlation_id: str) -> List[LogEntry]:
        """Get all entries for a correlation ID, sorted by timestamp."""
        entries = self.correlation_map.get(correlation_id, [])
        return sorted(entries, key=lambda x: x.timestamp)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from logs."""
        inference_entries = self.filter_by_event_type('inference')
        request_entries = self.filter_by_event_type('http_request')
        
        # Calculate inference metrics
        inference_durations = []
        for entry in inference_entries:
            if 'duration_ms' in entry.fields:
                inference_durations.append(entry.fields['duration_ms'])
        
        # Calculate request metrics
        request_durations = []
        for entry in request_entries:
            if 'duration_ms' in entry.fields:
                request_durations.append(entry.fields['duration_ms'])
        
        # Calculate statistics
        def calculate_stats(values):
            if not values:
                return {}
            values.sort()
            n = len(values)
            return {
                'count': n,
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / n,
                'p50': values[n // 2],
                'p95': values[int(n * 0.95)],
                'p99': values[int(n * 0.99)]
            }
        
        return {
            'inference': calculate_stats(inference_durations),
            'requests': calculate_stats(request_durations),
            'total_entries': len(self.entries),
            'error_count': len(self.filter_by_level('ERROR')) + len(self.filter_by_level('CRITICAL')),
            'event_counts': dict(self.event_counts),
            'error_counts': dict(self.error_counts)
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis from logs."""
        error_entries = self.filter_by_level('ERROR') + self.filter_by_level('CRITICAL')
        
        # Group errors by type
        error_types = defaultdict(list)
        for entry in error_entries:
            error_type = entry.fields.get('error_type', 'Unknown')
            error_types[error_type].append(entry)
        
        # Analyze error patterns
        error_analysis = {}
        for error_type, entries in error_types.items():
            error_analysis[error_type] = {
                'count': len(entries),
                'first_occurrence': min(entry.timestamp for entry in entries),
                'last_occurrence': max(entry.timestamp for entry in entries),
                'endpoints': list(set(entry.fields.get('endpoint', 'unknown') for entry in entries)),
                'sample_messages': [entry.message for entry in entries[:3]]
            }
        
        return {
            'total_errors': len(error_entries),
            'error_types': error_analysis,
            'most_common_errors': self.error_counts.most_common(10)
        }
    
    def get_correlation_analysis(self) -> Dict[str, Any]:
        """Get correlation analysis for request tracing."""
        correlation_stats = {}
        
        for correlation_id, entries in self.correlation_map.items():
            if len(entries) > 1:  # Only analyze multi-step traces
                durations = []
                event_types = []
                
                for entry in entries:
                    if 'duration_ms' in entry.fields:
                        durations.append(entry.fields['duration_ms'])
                    if entry.event_type:
                        event_types.append(entry.event_type)
                
                correlation_stats[correlation_id] = {
                    'entry_count': len(entries),
                    'total_duration_ms': sum(durations) if durations else 0,
                    'event_types': event_types,
                    'start_time': min(entry.timestamp for entry in entries),
                    'end_time': max(entry.timestamp for entry in entries)
                }
        
        return {
            'total_correlations': len(self.correlation_map),
            'multi_step_correlations': len(correlation_stats),
            'correlation_stats': correlation_stats
        }
    
    def export_to_json(self, file_path: str):
        """Export aggregated logs to JSON file."""
        data = {
            'summary': {
                'total_entries': len(self.entries),
                'time_range': {
                    'start': min(entry.timestamp for entry in self.entries).isoformat() if self.entries else None,
                    'end': max(entry.timestamp for entry in self.entries).isoformat() if self.entries else None
                },
                'performance': self.get_performance_summary(),
                'errors': self.get_error_analysis(),
                'correlations': self.get_correlation_analysis()
            },
            'entries': [
                {
                    'timestamp': entry.timestamp.isoformat(),
                    'level': entry.level,
                    'message': entry.message,
                    'correlation_id': entry.correlation_id,
                    'logger': entry.logger,
                    'event_type': entry.event_type,
                    'fields': entry.fields
                }
                for entry in self.entries
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def print_summary(self):
        """Print a summary of the aggregated logs."""
        summary = self.get_performance_summary()
        errors = self.get_error_analysis()
        correlations = self.get_correlation_analysis()
        
        print("=== CarlaRL Log Analysis Summary ===")
        print(f"Total log entries: {summary['total_entries']}")
        print(f"Error count: {summary['error_count']}")
        print(f"Total correlations: {correlations['total_correlations']}")
        print()
        
        print("=== Performance Metrics ===")
        if summary['inference']:
            print("Inference (ms):")
            for metric, value in summary['inference'].items():
                print(f"  {metric}: {value:.2f}")
        
        if summary['requests']:
            print("Requests (ms):")
            for metric, value in summary['requests'].items():
                print(f"  {metric}: {value:.2f}")
        
        print()
        print("=== Event Types ===")
        for event_type, count in summary['event_counts'].items():
            print(f"  {event_type}: {count}")
        
        print()
        print("=== Error Analysis ===")
        for error_type, analysis in errors['error_types'].items():
            print(f"  {error_type}: {analysis['count']} occurrences")
        
        print()
        print("=== Most Common Errors ===")
        for error, count in errors['most_common_errors']:
            print(f"  {error}: {count}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='CarlaRL Log Aggregator')
    parser.add_argument('--file', '-f', help='Log file to analyze')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--level', help='Filter by log level')
    parser.add_argument('--event-type', help='Filter by event type')
    parser.add_argument('--correlation-id', help='Filter by correlation ID')
    parser.add_argument('--search', help='Search by message pattern')
    parser.add_argument('--time-start', help='Start time (ISO format)')
    parser.add_argument('--time-end', help='End time (ISO format)')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    
    args = parser.parse_args()
    
    aggregator = LogAggregator()
    
    # Load logs
    if args.file:
        aggregator.load_from_file(args.file)
    else:
        aggregator.load_from_stdin()
    
    # Apply filters
    entries = aggregator.entries
    
    if args.level:
        entries = [e for e in entries if e.level == args.level]
    
    if args.event_type:
        entries = [e for e in entries if e.event_type == args.event_type]
    
    if args.correlation_id:
        entries = aggregator.filter_by_correlation_id(args.correlation_id)
    
    if args.search:
        entries = [e for e in entries if re.search(args.search, e.message, re.IGNORECASE)]
    
    if args.time_start:
        start_time = datetime.fromisoformat(args.time_start)
        entries = [e for e in entries if e.timestamp >= start_time]
    
    if args.time_end:
        end_time = datetime.fromisoformat(args.time_end)
        entries = [e for e in entries if e.timestamp <= end_time]
    
    # Output results
    if args.summary:
        aggregator.print_summary()
    elif args.output:
        aggregator.export_to_json(args.output)
    else:
        for entry in entries:
            print(entry.raw_line.rstrip())


if __name__ == '__main__':
    main()
