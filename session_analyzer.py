"""
Session Analyzer - ML-based pattern recognition for drowsiness sessions
Provides insights, recommendations, and risk assessment
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os


class SessionAnalyzer:
    """
    Analyzes session data to identify patterns and provide insights
    """

    def __init__(self):
        """Initialize analyzer"""
        self.scaler = StandardScaler()

    def load_session(self, session_file):
        """Load session data from JSON file"""
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        return session_data

    def analyze_session(self, session_data):
        """
        Comprehensive session analysis
        Returns insights, risk assessment, and recommendations
        """
        metrics_history = session_data.get('metrics_history', [])

        if len(metrics_history) == 0:
            return None

        df = pd.DataFrame(metrics_history)

        # Basic statistics
        basic_stats = {
            'session_duration': df['elapsed_time'].max(),
            'avg_fatigue': df['fatigue_score'].mean(),
            'max_fatigue': df['fatigue_score'].max(),
            'std_fatigue': df['fatigue_score'].std(),
            'avg_perclos': df['perclos'].mean(),
            'max_perclos': df['perclos'].max(),
            'avg_ear': df['ear'].mean(),
            'min_ear': df['ear'].min(),
            'total_blinks': df['blink_count'].max(),
            'total_yawns': df['yawn_count'].max(),
            'total_microsleeps': df['microsleep_count'].max(),
            'avg_blink_rate': df['blink_rate'].mean(),
            'avg_yawn_rate': df['yawn_rate'].mean()
        }

        # Time-based analysis
        time_analysis = self._analyze_temporal_patterns(df)

        # Alert analysis
        alert_analysis = self._analyze_alerts(df)

        # Risk assessment
        risk_assessment = self._assess_risk(basic_stats, alert_analysis)

        # Recommendations
        recommendations = self._generate_recommendations(
            basic_stats, time_analysis, alert_analysis, risk_assessment
        )

        # Fatigue patterns
        patterns = self._identify_patterns(df)

        return {
            'basic_statistics': basic_stats,
            'temporal_analysis': time_analysis,
            'alert_analysis': alert_analysis,
            'risk_assessment': risk_assessment,
            'recommendations': recommendations,
            'fatigue_patterns': patterns,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _analyze_temporal_patterns(self, df):
        """Analyze fatigue patterns over time"""
        # Split session into segments (5-minute intervals)
        segment_duration = 300  # 5 minutes
        max_time = df['elapsed_time'].max()
        num_segments = int(max_time / segment_duration) + 1

        segments = []
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration

            segment_df = df[
                (df['elapsed_time'] >= start_time) &
                (df['elapsed_time'] < end_time)
            ]

            if len(segment_df) > 0:
                segments.append({
                    'segment': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'avg_fatigue': segment_df['fatigue_score'].mean(),
                    'max_fatigue': segment_df['fatigue_score'].max(),
                    'avg_perclos': segment_df['perclos'].mean(),
                    'yawn_count': segment_df['yawn_count'].diff().sum()
                })

        # Identify trend
        if len(segments) > 1:
            fatigue_trend = [s['avg_fatigue'] for s in segments]
            trend_direction = 'increasing' if fatigue_trend[-1] > fatigue_trend[0] else 'decreasing'
            trend_rate = (fatigue_trend[-1] - fatigue_trend[0]) / len(segments)
        else:
            trend_direction = 'stable'
            trend_rate = 0

        return {
            'segments': segments,
            'trend_direction': trend_direction,
            'trend_rate': trend_rate,
            'most_fatigued_segment': max(segments, key=lambda x: x['avg_fatigue']) if segments else None
        }

    def _analyze_alerts(self, df):
        """Analyze alert distribution and patterns"""
        alert_counts = df['alert_level'].value_counts().to_dict()

        # Calculate alert duration percentages
        total_frames = len(df)
        alert_percentages = {
            level: (count / total_frames * 100)
            for level, count in alert_counts.items()
        }

        # Find alert episodes (consecutive frames with alert >= 2)
        df['is_alert'] = df['alert_level'] >= 2
        df['alert_episode'] = (df['is_alert'] != df['is_alert'].shift()).cumsum()

        alert_episodes = df[df['is_alert']].groupby('alert_episode').agg({
            'elapsed_time': ['min', 'max', 'count'],
            'alert_level': 'max'
        }).reset_index()

        if len(alert_episodes) > 0:
            alert_episodes.columns = ['episode', 'start_time', 'end_time', 'duration', 'max_level']
            alert_episodes['duration_seconds'] = (alert_episodes['end_time'] - alert_episodes['start_time'])
            num_episodes = len(alert_episodes)
            avg_episode_duration = alert_episodes['duration_seconds'].mean()
            longest_episode = alert_episodes['duration_seconds'].max()
        else:
            num_episodes = 0
            avg_episode_duration = 0
            longest_episode = 0

        return {
            'alert_counts': alert_counts,
            'alert_percentages': alert_percentages,
            'num_alert_episodes': num_episodes,
            'avg_episode_duration': avg_episode_duration,
            'longest_episode_duration': longest_episode,
            'critical_alerts': alert_counts.get(4, 0),
            'high_alerts': alert_counts.get(3, 0)
        }

    def _assess_risk(self, basic_stats, alert_analysis):
        """
        Assess overall drowsiness risk level
        Returns: low, moderate, high, critical
        """
        risk_score = 0

        # Factor 1: Average fatigue
        if basic_stats['avg_fatigue'] > 70:
            risk_score += 4
        elif basic_stats['avg_fatigue'] > 50:
            risk_score += 3
        elif basic_stats['avg_fatigue'] > 30:
            risk_score += 2
        elif basic_stats['avg_fatigue'] > 15:
            risk_score += 1

        # Factor 2: Microsleep episodes
        if basic_stats['total_microsleeps'] > 5:
            risk_score += 4
        elif basic_stats['total_microsleeps'] > 2:
            risk_score += 2
        elif basic_stats['total_microsleeps'] > 0:
            risk_score += 1

        # Factor 3: Critical alerts
        if alert_analysis['critical_alerts'] > 10:
            risk_score += 3
        elif alert_analysis['critical_alerts'] > 5:
            risk_score += 2
        elif alert_analysis['critical_alerts'] > 0:
            risk_score += 1

        # Factor 4: PERCLOS
        if basic_stats['avg_perclos'] > 30:
            risk_score += 3
        elif basic_stats['avg_perclos'] > 20:
            risk_score += 2
        elif basic_stats['avg_perclos'] > 10:
            risk_score += 1

        # Factor 5: Yawn rate
        if basic_stats['avg_yawn_rate'] > 6:
            risk_score += 2
        elif basic_stats['avg_yawn_rate'] > 3:
            risk_score += 1

        # Determine risk level
        if risk_score >= 12:
            risk_level = 'critical'
            risk_description = 'Severe drowsiness detected. Immediate action required.'
        elif risk_score >= 8:
            risk_level = 'high'
            risk_description = 'High risk of drowsiness-related incidents.'
        elif risk_score >= 4:
            risk_level = 'moderate'
            risk_description = 'Moderate fatigue levels detected.'
        else:
            risk_level = 'low'
            risk_description = 'Normal alertness levels maintained.'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_description': risk_description
        }

    def _generate_recommendations(self, basic_stats, time_analysis, alert_analysis, risk_assessment):
        """Generate personalized recommendations"""
        recommendations = []

        # Risk-based recommendations
        if risk_assessment['risk_level'] in ['critical', 'high']:
            recommendations.append({
                'priority': 'critical',
                'category': 'immediate_action',
                'recommendation': 'Stop all activities and take an immediate rest break of at least 20-30 minutes.',
                'reason': f"Risk level: {risk_assessment['risk_level']}"
            })

        # Microsleep recommendations
        if basic_stats['total_microsleeps'] > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'sleep',
                'recommendation': f"You experienced {basic_stats['total_microsleeps']} microsleep episodes. This indicates severe sleep deprivation. Plan for 7-8 hours of sleep tonight.",
                'reason': 'Microsleep episodes detected'
            })

        # PERCLOS recommendations
        if basic_stats['avg_perclos'] > 20:
            recommendations.append({
                'priority': 'high',
                'category': 'rest',
                'recommendation': 'Your eyes were closed more than 20% of the time. Take frequent breaks every 15-20 minutes.',
                'reason': f"High PERCLOS: {basic_stats['avg_perclos']:.1f}%"
            })

        # Yawning recommendations
        if basic_stats['avg_yawn_rate'] > 3:
            recommendations.append({
                'priority': 'medium',
                'category': 'environment',
                'recommendation': 'Frequent yawning detected. Improve ventilation, reduce temperature, or take a short walk.',
                'reason': f"High yawn rate: {basic_stats['avg_yawn_rate']:.1f}/min"
            })

        # Blink rate recommendations
        if basic_stats['avg_blink_rate'] < 10:
            recommendations.append({
                'priority': 'medium',
                'category': 'eye_health',
                'recommendation': 'Low blink rate detected. Practice the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds.',
                'reason': f"Reduced blink rate: {basic_stats['avg_blink_rate']:.1f}/min"
            })

        # Trend-based recommendations
        if time_analysis['trend_direction'] == 'increasing':
            recommendations.append({
                'priority': 'medium',
                'category': 'scheduling',
                'recommendation': 'Your fatigue increased steadily throughout the session. Consider taking breaks earlier in your next session.',
                'reason': 'Increasing fatigue trend'
            })

        # Session duration recommendations
        if basic_stats['session_duration'] > 3600:  # More than 1 hour
            recommendations.append({
                'priority': 'low',
                'category': 'best_practices',
                'recommendation': 'For sessions longer than 1 hour, plan mandatory breaks every 30-45 minutes.',
                'reason': f"Long session duration: {basic_stats['session_duration']/60:.0f} minutes"
            })

        # Positive reinforcement
        if risk_assessment['risk_level'] == 'low' and basic_stats['session_duration'] > 600:
            recommendations.append({
                'priority': 'info',
                'category': 'positive',
                'recommendation': 'Good job! You maintained healthy alertness levels throughout the session.',
                'reason': 'Low risk profile'
            })

        return recommendations

    def _identify_patterns(self, df):
        """Use clustering to identify fatigue patterns"""
        # Select features for clustering
        features = ['fatigue_score', 'perclos', 'ear', 'blink_rate', 'yawn_rate']

        # Prepare data
        X = df[features].dropna()

        if len(X) < 10:
            return None

        # Normalize
        X_scaled = self.scaler.fit_transform(X)

        # Clustering (3 clusters: low, medium, high fatigue)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Map clusters to fatigue levels
        cluster_means = []
        for i in range(3):
            cluster_data = df.iloc[X.index[clusters == i]]
            cluster_means.append({
                'cluster': i,
                'avg_fatigue': cluster_data['fatigue_score'].mean(),
                'count': len(cluster_data)
            })

        # Sort by fatigue level
        cluster_means.sort(key=lambda x: x['avg_fatigue'])

        # Label clusters
        labels = ['low_fatigue', 'medium_fatigue', 'high_fatigue']
        for i, cm in enumerate(cluster_means):
            cm['label'] = labels[i]

        return {
            'clusters': cluster_means,
            'distribution': {
                'low_fatigue_percent': cluster_means[0]['count'] / len(X) * 100,
                'medium_fatigue_percent': cluster_means[1]['count'] / len(X) * 100,
                'high_fatigue_percent': cluster_means[2]['count'] / len(X) * 100
            }
        }

    def export_report(self, session_file, output_format='txt'):
        """
        Export comprehensive analysis report
        Formats: txt, json, html
        """
        session_data = self.load_session(session_file)
        analysis = self.analyze_session(session_data)

        if not analysis:
            return None

        session_id = session_data.get('session_id', 'unknown')
        output_dir = os.path.dirname(session_file)

        if output_format == 'json':
            output_file = os.path.join(output_dir, f'analysis_{session_id}.json')
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)

        elif output_format == 'txt':
            output_file = os.path.join(output_dir, f'report_{session_id}.txt')
            report = self._generate_text_report(session_data, analysis)
            with open(output_file, 'w') as f:
                f.write(report)

        return output_file

    def _generate_text_report(self, session_data, analysis):
        """Generate human-readable text report"""
        report = []
        report.append("=" * 80)
        report.append("DROWSINESS DETECTION SESSION REPORT")
        report.append("=" * 80)
        report.append("")

        # Session info
        report.append(f"Session ID: {session_data.get('session_id', 'N/A')}")
        report.append(f"Start Time: {session_data.get('start_time', 'N/A')}")
        report.append(f"End Time: {session_data.get('end_time', 'N/A')}")
        report.append("")

        # Risk assessment
        risk = analysis['risk_assessment']
        report.append("-" * 80)
        report.append("RISK ASSESSMENT")
        report.append("-" * 80)
        report.append(f"Risk Level: {risk['risk_level'].upper()}")
        report.append(f"Risk Score: {risk['risk_score']}/15")
        report.append(f"Description: {risk['risk_description']}")
        report.append("")

        # Basic statistics
        stats = analysis['basic_statistics']
        report.append("-" * 80)
        report.append("SESSION STATISTICS")
        report.append("-" * 80)
        report.append(f"Duration: {stats['session_duration']/60:.1f} minutes")
        report.append(f"Average Fatigue Score: {stats['avg_fatigue']:.1f}%")
        report.append(f"Maximum Fatigue Score: {stats['max_fatigue']:.1f}%")
        report.append(f"Average PERCLOS: {stats['avg_perclos']:.1f}%")
        report.append(f"Total Blinks: {stats['total_blinks']}")
        report.append(f"Total Yawns: {stats['total_yawns']}")
        report.append(f"Microsleep Episodes: {stats['total_microsleeps']}")
        report.append(f"Average Blink Rate: {stats['avg_blink_rate']:.1f}/min")
        report.append(f"Average Yawn Rate: {stats['avg_yawn_rate']:.1f}/min")
        report.append("")

        # Alert analysis
        alerts = analysis['alert_analysis']
        report.append("-" * 80)
        report.append("ALERT SUMMARY")
        report.append("-" * 80)
        report.append(f"Critical Alerts (Level 4): {alerts['critical_alerts']}")
        report.append(f"High Alerts (Level 3): {alerts['high_alerts']}")
        report.append(f"Alert Episodes: {alerts['num_alert_episodes']}")
        if alerts['num_alert_episodes'] > 0:
            report.append(f"Longest Alert Episode: {alerts['longest_episode_duration']:.1f}s")
        report.append("")

        # Recommendations
        report.append("-" * 80)
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        for i, rec in enumerate(analysis['recommendations'], 1):
            report.append(f"{i}. [{rec['priority'].upper()}] {rec['recommendation']}")
            report.append(f"   Reason: {rec['reason']}")
            report.append("")

        report.append("=" * 80)
        report.append(f"Report generated: {analysis['analysis_timestamp']}")
        report.append("=" * 80)

        return "\n".join(report)


# CLI for analyzing sessions
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python session_analyzer.py <session_json_file> [output_format]")
        print("Output formats: txt (default), json")
        sys.exit(1)

    session_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'txt'

    analyzer = SessionAnalyzer()
    output_file = analyzer.export_report(session_file, output_format)

    if output_file:
        print(f"Analysis report generated: {output_file}")
    else:
        print("Failed to generate report")
