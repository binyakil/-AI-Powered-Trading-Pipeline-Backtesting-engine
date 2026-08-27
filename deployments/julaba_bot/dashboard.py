# Performance Dashboard for Julaba
# Simple Flask-based web dashboard for monitoring trading performance.

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import asdict
import json
import threading

logger = logging.getLogger("Julaba.Dashboard")

# Dashboard dependencies (optional)
try:
    from flask import Flask, render_template_string, jsonify, request, redirect, make_response
    from flask.json.provider import DefaultJSONProvider
    import numpy as np
    
    # Custom JSON provider to handle numpy types
    class NumpyJSONProvider(DefaultJSONProvider):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)
    
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not installed. Run: pip install flask")


# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=1400, initial-scale=0.5, maximum-scale=2.0, user-scalable=yes">
    <meta name="HandheldFriendly" content="false">
    <meta name="MobileOptimized" content="1400">
    <title>Julaba Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1/dist/chartjs-chart-financial.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        /* News headline hover styles */
        .news-headline-item { background: rgba(255,255,255,0.03); transition: background 0.2s; }
        .news-headline-item:hover { background: rgba(255,255,255,0.07); }
        .news-headline-link { color: #ccc; text-decoration: none; }
        .news-headline-link:hover { color: #00d4ff; }
        
        /* ========== ROOT VARIABLES ========== */
        :root {
            --primary: #00d4ff;
            --accent: #00ff88;
            --danger: #ff4444;
            --warning: #ffaa00;
            --bg-card: rgba(255, 255, 255, 0.06);
            --bg-card-hover: rgba(255, 255, 255, 0.10);
            --border-subtle: rgba(0, 212, 255, 0.15);
            --border-hover: rgba(0, 212, 255, 0.4);
            --text-primary: #eee;
            --text-secondary: #aaa;
            --text-muted: #666;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.25);
            --shadow-hover: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 212, 255, 0.15);
        }

        /* Animated Background */
        body {
            font-family: 'Rajdhani', 'Segoe UI', sans-serif;
            background: #050510;
            color: var(--text-primary);
            min-height: 100vh;
            min-width: 1400px;
            overflow-x: auto;
            position: relative;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            line-height: 1.5;
        }
        
        /* Global scrollbar styling */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); border-radius: 4px; }
        ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.35); }

        #bg-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        
        .main-content {
            position: relative;
            z-index: 1;
            padding: 20px 24px;
        }
        
        .container { max-width: 1600px; margin: 0 auto; }
        
        /* 3D Title */
        h1 {
            text-align: center;
            margin-bottom: 40px;
            font-family: 'Orbitron', sans-serif;
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00d4ff, #00ff88, #00d4ff);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 3s ease infinite;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
            filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.5));
            transform: perspective(500px) rotateX(5deg);
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .section-title {
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff;
            font-size: 1.1rem;
            margin: 25px 0 15px 0;
            padding-bottom: 10px;
            padding-left: 12px;
            border-bottom: 2px solid transparent;
            border-left: 3px solid rgba(0, 212, 255, 0.6);
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.5), transparent);
            background-size: 100% 2px;
            background-position: bottom;
            background-repeat: no-repeat;
            text-transform: uppercase;
            letter-spacing: 2.5px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
            perspective: 1000px;
        }
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
            margin-bottom: 24px;
            perspective: 1000px;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
            margin-bottom: 24px;
            perspective: 1000px;
        }
        /* Desktop mode forced on all devices - no mobile layout */
        
        /* 3D Glass Cards */
        .card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
            border-radius: var(--radius-md);
            padding: 16px 18px;
            border: 1px solid var(--border-subtle);
            backdrop-filter: blur(20px);
            transform-style: preserve-3d;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-card);
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.15), transparent);
            transition: left 0.5s;
        }
        
        .card:hover {
            transform: translateY(-6px) rotateX(3deg) rotateY(-1deg) scale(1.01);
            border-color: var(--border-hover);
            box-shadow: var(--shadow-hover);
        }
        
        .card:hover::before {
            left: 100%;
        }
        
        .card-lg {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.01));
            border-radius: var(--radius-lg);
            padding: 18px 22px;
            border: 1px solid var(--border-subtle);
            backdrop-filter: blur(20px);
            transform-style: preserve-3d;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-card);
        }
        
        /* THE MACHINE - Fully transparent to show stars/galaxy */
        .card-lg.machine-section {
            background: transparent !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            border: 1px solid rgba(0, 212, 255, 0.4);
            box-shadow: 
                0 0 40px rgba(0, 212, 255, 0.15),
                inset 0 0 60px rgba(0, 212, 255, 0.05);
        }
        .card-lg.machine-section:hover {
            transform: none;
            border-color: rgba(0, 212, 255, 0.6);
            box-shadow: 
                0 0 60px rgba(0, 212, 255, 0.25),
                inset 0 0 80px rgba(0, 212, 255, 0.08);
        }
        
        /* PIPELINE MONITOR - Ship Engine Room Style */
        .pipeline-central {
            background: linear-gradient(135deg, rgba(10, 15, 25, 0.98) 0%, rgba(20, 25, 40, 0.98) 100%);
            border: 3px solid rgba(255, 136, 0, 0.5);
            box-shadow: 0 0 50px rgba(255, 136, 0, 0.3), inset 0 0 80px rgba(255, 136, 0, 0.08);
            position: relative;
            overflow: hidden;
        }
        .pipeline-central::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 200%;
            height: 3px;
            background: linear-gradient(90deg, transparent, #ff8800, #00d4ff, transparent);
            animation: scanLine 3s linear infinite;
        }
        @keyframes scanLine {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        .pipeline-component {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 18px 22px;
            background: linear-gradient(180deg, rgba(30, 35, 55, 0.95) 0%, rgba(20, 25, 40, 0.95) 100%);
            border: 2px solid #444;
            border-radius: 12px;
            min-width: 120px;
            position: relative;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .pipeline-component::before {
            content: '';
            position: absolute;
            top: -2px;
            left: 50%;
            transform: translateX(-50%);
            width: 40%;
            height: 3px;
            background: linear-gradient(90deg, transparent, currentColor, transparent);
            opacity: 0.5;
        }
        .pipeline-component.processing {
            animation: processingGlow 0.5s ease-in-out infinite alternate;
        }
        @keyframes processingGlow {
            from { box-shadow: 0 0 20px rgba(0, 212, 255, 0.4); }
            to { box-shadow: 0 0 40px rgba(0, 212, 255, 0.8), 0 0 60px rgba(0, 212, 255, 0.4); }
        }
        .pipeline-component:hover {
            transform: scale(1.08);
            z-index: 10;
            border-color: #00d4ff;
        }
        .pipeline-component.status-ok {
            border-color: #00ff88;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.4), inset 0 0 30px rgba(0, 255, 136, 0.08);
        }
        .pipeline-component.status-ok::before { color: #00ff88; }
        .pipeline-component.status-warning {
            border-color: #ffcc00;
            box-shadow: 0 0 20px rgba(255, 204, 0, 0.4), inset 0 0 30px rgba(255, 204, 0, 0.08);
            animation: warningPulse 1.5s ease-in-out infinite;
        }
        .pipeline-component.status-warning::before { color: #ffcc00; }
        .pipeline-component.status-error {
            border-color: #ff4444;
            box-shadow: 0 0 25px rgba(255, 68, 68, 0.6), inset 0 0 40px rgba(255, 68, 68, 0.12);
            animation: errorPulse 0.8s ease-in-out infinite;
        }
        .pipeline-component.status-error::before { color: #ff4444; }
        .pipeline-component.status-cooldown {
            border-color: #aa00ff;
            box-shadow: 0 0 20px rgba(170, 0, 255, 0.4);
        }
        .pipeline-component.status-cooldown::before { color: #aa00ff; }
        .pipeline-component.status-blocked {
            border-color: #ff8800;
            box-shadow: 0 0 20px rgba(255, 136, 0, 0.4);
        }
        .pipeline-component.status-active {
            border-color: #00d4ff;
            box-shadow: 0 0 25px rgba(0, 212, 255, 0.5);
            animation: activePulse 1s ease-in-out infinite;
        }
        .pipeline-component.status-idle {
            border-color: #555;
            opacity: 0.6;
        }
        .pipeline-icon {
            font-size: 2rem;
            margin-bottom: 8px;
            filter: drop-shadow(0 0 5px currentColor);
        }
        .pipeline-name {
            font-size: 0.7rem;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Orbitron', sans-serif;
        }
        .pipeline-status-indicator {
            position: absolute;
            top: 8px;
            right: 8px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff88;
            box-shadow: 0 0 10px currentColor;
        }
        .pipeline-status-indicator.warning { background: #ffcc00; box-shadow: 0 0 10px #ffcc00; }
        .pipeline-status-indicator.error { background: #ff4444; box-shadow: 0 0 10px #ff4444; animation: blink 0.5s infinite; }
        .pipeline-status-indicator.cooldown { background: #aa00ff; box-shadow: 0 0 10px #aa00ff; }
        .pipeline-detail {
            font-size: 0.7rem;
            color: #00d4ff;
            margin-top: 5px;
            font-family: 'Orbitron', sans-serif;
            text-shadow: 0 0 5px rgba(0, 212, 255, 0.5);
        }
        .pipeline-connector {
            width: 50px;
            height: 4px;
            background: linear-gradient(90deg, #333, #00d4ff 30%, #00d4ff 70%, #333);
            align-self: center;
            position: relative;
            border-radius: 2px;
            overflow: hidden;
        }
        .pipeline-connector::before {
            content: '';
            position: absolute;
            top: 0;
            left: -50px;
            width: 30px;
            height: 100%;
            background: linear-gradient(90deg, transparent, #fff, transparent);
            animation: dataFlow 1s linear infinite;
        }
        @keyframes dataFlow {
            0% { left: -30px; }
            100% { left: 50px; }
        }
        .pipeline-connector::after {
            content: '▶';
            position: absolute;
            right: -8px;
            top: -6px;
            font-size: 12px;
            color: #00d4ff;
            text-shadow: 0 0 10px #00d4ff;
        }
        .pipeline-connector.inactive {
            background: linear-gradient(90deg, #222, #444 30%, #444 70%, #222);
        }
        .pipeline-connector.inactive::before {
            animation: none;
            display: none;
        }
        .pipeline-connector.inactive::after {
            color: #444;
            text-shadow: none;
        }
        /* Activity indicator */
        .pipeline-activity {
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.6rem;
            color: #00d4ff;
            white-space: nowrap;
            animation: fadeInOut 2s ease-in-out infinite;
        }
        @keyframes fadeInOut {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
        /* Pipeline component clickable */
        .pipeline-component {
            cursor: pointer;
        }
        .pipeline-component:hover {
            transform: scale(1.08);
            z-index: 10;
        }
        
        /* PIPELINE POPUP - Transparent Toggle */
        .pipeline-popup {
            display: none;
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            min-width: 280px;
            max-width: 350px;
            background: rgba(15, 20, 35, 0.95);
            border: 2px solid #00d4ff;
            border-radius: 12px;
            padding: 15px;
            z-index: 1000;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 30px rgba(0, 212, 255, 0.2);
            backdrop-filter: blur(10px);
            margin-top: 10px;
        }
        .pipeline-popup.active {
            display: block;
            animation: popupFadeIn 0.2s ease;
        }
        @keyframes popupFadeIn {
            from { opacity: 0; transform: translateX(-50%) translateY(-10px); }
            to { opacity: 1; transform: translateX(-50%) translateY(0); }
        }
        .pipeline-popup::before {
            content: '';
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-bottom: 8px solid #00d4ff;
        }
        .pipeline-popup-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            color: #00d4ff;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        .pipeline-popup-row {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 0.8rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .pipeline-popup-label { color: #888; }
        .pipeline-popup-value { color: #fff; font-family: 'Orbitron', sans-serif; }
        .pipeline-popup-value.ok { color: #00ff88; }
        .pipeline-popup-value.warning { color: #ffcc00; }
        .pipeline-popup-value.error { color: #ff4444; }
        .pipeline-popup-value.info { color: #00d4ff; }
        
        /* Component Status Bar */
        .comp-status {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            background: rgba(0,0,0,0.4);
            border: 1px solid #333;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .comp-status:hover {
            border-color: #00d4ff;
            background: rgba(0,212,255,0.1);
        }
        .comp-status.ok { border-color: rgba(0,255,136,0.5); }
        .comp-status.ok span { color: #00ff88; }
        .comp-status.warning { border-color: rgba(255,204,0,0.5); }
        .comp-status.warning span { color: #ffcc00; }
        .comp-status.error { border-color: rgba(255,68,68,0.5); background: rgba(255,68,68,0.1); animation: error-pulse 1s infinite; }
        .comp-status.error span { color: #ff4444; }
        .comp-status.disabled { border-color: rgba(100,100,100,0.5); opacity: 0.6; }
        .comp-status.disabled span { color: #666; }
        .comp-status.initializing { border-color: rgba(0,212,255,0.5); }
        .comp-status.initializing span { color: #00d4ff; animation: blink 0.5s infinite; }
        .comp-status.stopped { border-color: rgba(255,68,68,0.8); background: rgba(255,68,68,0.2); }
        .comp-status.stopped span { color: #ff4444; font-weight: bold; }
        
        /* Engine status - prominent styling */
        .comp-status.engine-status {
            background: linear-gradient(135deg, rgba(0,20,40,0.8), rgba(0,40,60,0.6));
            border: 1px solid #00d4ff;
            box-shadow: 0 0 8px rgba(0,212,255,0.3);
        }
        .comp-status.engine-status.ok {
            border-color: #00ff88;
            box-shadow: 0 0 10px rgba(0,255,136,0.4);
        }
        .comp-status.engine-status.stopped {
            border-color: #ff4444;
            box-shadow: 0 0 15px rgba(255,68,68,0.5);
            animation: error-glow 1s infinite;
        }
        @keyframes error-glow {
            0%, 100% { box-shadow: 0 0 10px rgba(255,68,68,0.5); }
            50% { box-shadow: 0 0 20px rgba(255,68,68,0.8); }
        }
        
        /* Divider between engine and other components */
        .comp-status-divider {
            color: #444;
            font-size: 0.8rem;
            padding: 0 2px;
            align-self: center;
        }
        
        /* Pipeline Rows Layout - deprecated but keep for compatibility */
        .pipeline-row {
            display: flex;
            gap: 8px;
            justify-content: center;
            align-items: center;
            padding: 10px 0;
        }
        .pipeline-row-label {
            font-size: 0.65rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            width: 80px;
            text-align: right;
            padding-right: 15px;
        }
        .pipeline-row-connector {
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
            margin: 5px 0;
        }
        
        .pipeline-overlay-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        .pipeline-overlay-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.2rem;
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .pipeline-overlay-close {
            font-size: 1.5rem;
            color: #888;
            cursor: pointer;
            transition: color 0.2s;
            background: none;
            border: none;
            padding: 5px 10px;
        }
        .pipeline-overlay-close:hover {
            color: #ff4444;
        }
        .pipeline-overlay-body {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .pipeline-detail-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            border-left: 3px solid #00d4ff;
        }
        .pipeline-detail-label {
            color: #888;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .pipeline-detail-value {
            color: #fff;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.95rem;
        }
        .pipeline-detail-value.positive { color: #00ff88; }
        .pipeline-detail-value.negative { color: #ff4444; }
        .pipeline-detail-value.warning { color: #ffcc00; }
        .pipeline-detail-value.info { color: #00d4ff; }
        .popup-section-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            color: #00d4ff;
            margin-top: 15px;
            margin-bottom: 8px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .popup-section-title:first-child {
            margin-top: 0;
        }
        @keyframes overlayFadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes overlaySlideUp {
            from { transform: translateY(30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes warningPulse {
            0%, 100% { box-shadow: 0 0 15px rgba(255, 204, 0, 0.3); }
            50% { box-shadow: 0 0 25px rgba(255, 204, 0, 0.6); }
        }
        @keyframes errorPulse {
            0%, 100% { box-shadow: 0 0 20px rgba(255, 68, 68, 0.5); }
            50% { box-shadow: 0 0 35px rgba(255, 68, 68, 0.8); }
        }
        @keyframes activePulse {
            0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.4); }
            50% { box-shadow: 0 0 30px rgba(0, 212, 255, 0.7); }
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        @keyframes flowArrow {
            0%, 100% { opacity: 0.5; transform: translateX(0); }
            50% { opacity: 1; transform: translateX(3px); }
        }
        
        /* ═══════════════════════════════════════════════════════════════ */
        /* 🌊 PIPELINE WATER FLOW EFFECTS                                   */
        /* ═══════════════════════════════════════════════════════════════ */
        @keyframes waterFlowPulse {
            0% { opacity: 0.3; box-shadow: 0 0 10px rgba(0,212,255,0.2); }
            50% { opacity: 1; box-shadow: 0 0 30px rgba(0,255,136,0.6), 0 0 60px rgba(0,212,255,0.4); }
            100% { opacity: 0.3; box-shadow: 0 0 10px rgba(0,212,255,0.2); }
        }
        
        @keyframes positionActiveGlow {
            0% { box-shadow: inset 0 0 30px rgba(0,255,136,0.1), 0 0 40px rgba(0,255,136,0.2); }
            50% { box-shadow: inset 0 0 50px rgba(0,255,136,0.2), 0 0 60px rgba(0,255,136,0.4), 0 0 100px rgba(0,212,255,0.2); }
            100% { box-shadow: inset 0 0 30px rgba(0,255,136,0.1), 0 0 40px rgba(0,255,136,0.2); }
        }
        
        @keyframes positionShortGlow {
            0% { box-shadow: inset 0 0 30px rgba(255,68,68,0.1), 0 0 40px rgba(255,68,68,0.2); }
            50% { box-shadow: inset 0 0 50px rgba(255,68,68,0.2), 0 0 60px rgba(255,68,68,0.4), 0 0 100px rgba(255,136,0,0.2); }
            100% { box-shadow: inset 0 0 30px rgba(255,68,68,0.1), 0 0 40px rgba(255,68,68,0.2); }
        }
        
        .machine-section.position-active {
            animation: positionActiveGlow 2s ease-in-out infinite;
            border-color: rgba(0, 255, 136, 0.5) !important;
        }
        
        .machine-section.position-short-active {
            animation: positionShortGlow 2s ease-in-out infinite;
            border-color: rgba(255, 68, 68, 0.5) !important;
        }
        
        .pipeline-flow-indicator {
            position: absolute;
            bottom: 8px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 15px;
            background: rgba(0,20,40,0.9);
            border-radius: 20px;
            border: 1px solid rgba(0,212,255,0.3);
            font-size: 11px;
            font-family: 'Orbitron', sans-serif;
            opacity: 1;
            transition: all 0.5s;
            z-index: 100;
        }
        
        .pipeline-flow-indicator.position-active {
            border-color: rgba(0,255,136,0.6);
            box-shadow: 0 0 15px rgba(0,255,136,0.3);
        }
        
        .pipeline-flow-indicator.position-short {
            border-color: rgba(255,68,68,0.6);
            box-shadow: 0 0 15px rgba(255,68,68,0.3);
        }
        
        .flow-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff88;
            animation: flowDotPulse 0.8s ease-in-out infinite;
        }
        
        .flow-dot.scanning {
            background: #00d4ff;
        }
        
        .flow-dot.short {
            background: #ff4444;
        }
        
        @keyframes flowDotPulse {
            0%, 100% { transform: scale(1); opacity: 0.7; }
            50% { transform: scale(1.3); opacity: 1; }
        }
        
        .card-lg:hover {
            transform: translateY(-5px) scale(1.01);
            border-color: rgba(0, 212, 255, 0.4);
            box-shadow: 
                0 15px 40px rgba(0, 0, 0, 0.4),
                0 0 25px rgba(0, 212, 255, 0.2);
        }
        
        .card h3 {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.65rem;
            color: #999;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            white-space: nowrap;
        }
        
        .card .value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.4rem;
            font-weight: bold;
            color: #fff;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
            animation: valueGlow 2s ease-in-out infinite alternate;
            line-height: 1.2;
        }
        
        @keyframes valueGlow {
            from { text-shadow: 0 0 10px rgba(255, 255, 255, 0.3); }
            to { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5), 0 0 30px rgba(0, 212, 255, 0.3); }
        }
        
        .card .value.sm { font-size: 1rem; }
        .card .value.positive { color: #00ff88; text-shadow: 0 0 15px rgba(0, 255, 136, 0.5); }
        .card .value.negative { color: #ff4444; text-shadow: 0 0 15px rgba(255, 68, 68, 0.5); }
        .card .value.warning { color: #ffaa00; text-shadow: 0 0 15px rgba(255,170,0, 0.5); }
        .card .value.info { color: #00d4ff; text-shadow: 0 0 15px rgba(0, 212, 255, 0.5); }
        .card .sub { font-size: 0.75rem; color: #777; margin-top: 4px; font-family: 'Rajdhani', sans-serif; }
        
        .indicator-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 9px 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s;
            border-radius: 6px;
        }
        .indicator-row:hover {
            background: rgba(0, 212, 255, 0.06);
            padding-left: 10px;
        }
        .indicator-row:last-child { border-bottom: none; }
        .indicator-name { color: #aab; font-size: 0.85rem; letter-spacing: 0.3px; }
        .indicator-value { 
            font-family: 'Orbitron', sans-serif;
            font-weight: bold; 
            font-size: 0.95rem;
            text-align: right;
        }
        .indicator-bar {
            width: 100px;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-left: 10px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        }
        .indicator-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 0 10px currentColor;
        }
        
        /* 3D Signal Box */
        .signal-box {
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 15px;
            transform-style: preserve-3d;
            transition: all 0.4s;
            position: relative;
            overflow: hidden;
        }
        .signal-box::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            animation: signalShine 3s infinite;
        }
        @keyframes signalShine {
            0% { transform: translateX(-100%) rotate(45deg); }
            100% { transform: translateX(100%) rotate(45deg); }
        }
        .signal-box.long { 
            background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,255,136,0.05)); 
            border: 2px solid rgba(0,255,136,0.4);
            box-shadow: 0 0 30px rgba(0,255,136,0.2), inset 0 0 30px rgba(0,255,136,0.1);
        }
        .signal-box.short { 
            background: linear-gradient(135deg, rgba(255,68,68,0.2), rgba(255,68,68,0.05)); 
            border: 2px solid rgba(255,68,68,0.4);
            box-shadow: 0 0 30px rgba(255,68,68,0.2), inset 0 0 30px rgba(255,68,68,0.1);
        }
        .signal-box.neutral { 
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.02)); 
            border: 2px solid rgba(255,255,255,0.2);
        }
        .signal-label { font-size: 0.8rem; color: #888; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px; }
        .signal-value { 
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem; 
            font-weight: bold;
            text-shadow: 0 0 20px currentColor;
        }
        
        /* ===== ENHANCED LIVE TRADING VISUALIZATION ===== */
        
        /* Signal Pulse Animation - When active signal */
        .signal-box.active-signal {
            animation: activeSignalPulse 1.5s ease-in-out infinite;
        }
        @keyframes activeSignalPulse {
            0%, 100% { transform: scale(1); filter: brightness(1); }
            50% { transform: scale(1.02); filter: brightness(1.2); }
        }
        
        /* Live Trade Progress Bar */
        .trade-progress-container {
            background: linear-gradient(135deg, rgba(0,0,0,0.5), rgba(0,0,0,0.3));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            position: relative;
        }
        
        .trade-progress-bar {
            height: 40px;
            background: linear-gradient(90deg, rgba(255,68,68,0.2), rgba(100,100,100,0.1) 30%, rgba(100,100,100,0.1) 70%, rgba(0,255,136,0.2));
            border-radius: 8px;
            position: relative;
            overflow: visible;
            margin: 20px 0;
        }
        
        .trade-level {
            position: absolute;
            top: -25px;
            transform: translateX(-50%);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 4px;
            white-space: nowrap;
        }
        
        .trade-level-line {
            position: absolute;
            top: 0;
            width: 2px;
            height: 40px;
            transform: translateX(-50%);
        }
        
        .trade-level.sl { color: #ff4444; }
        .trade-level.sl .trade-level-line { background: linear-gradient(180deg, #ff4444, transparent); }
        
        .trade-level.entry { color: #00d4ff; }
        .trade-level.entry .trade-level-line { background: #00d4ff; box-shadow: 0 0 10px #00d4ff; }
        
        .trade-level.tp1 { color: #88ff88; }
        .trade-level.tp1 .trade-level-line { background: linear-gradient(180deg, #88ff88, transparent); }
        
        .trade-level.tp2 { color: #44ff44; }
        .trade-level.tp2 .trade-level-line { background: linear-gradient(180deg, #44ff44, transparent); }
        
        .trade-level.tp3 { color: #00ff88; }
        .trade-level.tp3 .trade-level-line { background: linear-gradient(180deg, #00ff88, transparent); }
        
        .trade-level-label {
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.55rem;
            color: #666;
            white-space: nowrap;
        }
        
        /* Current Price Marker */
        .current-price-marker {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            z-index: 10;
        }
        
        .price-dot {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #fff;
            box-shadow: 0 0 20px #fff, 0 0 40px #fff;
            animation: pricePulse 1s ease-in-out infinite;
        }
        
        .price-dot.profit { background: #00ff88; box-shadow: 0 0 20px #00ff88, 0 0 40px #00ff88; }
        .price-dot.loss { background: #ff4444; box-shadow: 0 0 20px #ff4444, 0 0 40px #ff4444; }
        
        @keyframes pricePulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.8; }
        }
        
        .current-price-label {
            position: absolute;
            top: -30px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            color: #000;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            font-weight: bold;
            padding: 4px 10px;
            border-radius: 6px;
            white-space: nowrap;
            animation: labelFloat 2s ease-in-out infinite;
        }
        
        @keyframes labelFloat {
            0%, 100% { transform: translateX(-50%) translateY(0); }
            50% { transform: translateX(-50%) translateY(-3px); }
        }
        
        /* Animated PnL Display */
        .pnl-live-display {
            text-align: center;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            margin-top: 10px;
        }
        
        .pnl-value-large {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .pnl-value-large.profit {
            color: #00ff88;
            text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
            animation: profitGlow 1.5s ease-in-out infinite;
        }
        
        .pnl-value-large.loss {
            color: #ff4444;
            text-shadow: 0 0 30px rgba(255, 68, 68, 0.5);
            animation: lossGlow 1.5s ease-in-out infinite;
        }
        
        @keyframes profitGlow {
            0%, 100% { text-shadow: 0 0 20px rgba(0, 255, 136, 0.5); }
            50% { text-shadow: 0 0 40px rgba(0, 255, 136, 0.8), 0 0 60px rgba(0, 255, 136, 0.4); }
        }
        
        @keyframes lossGlow {
            0%, 100% { text-shadow: 0 0 20px rgba(255, 68, 68, 0.5); }
            50% { text-shadow: 0 0 40px rgba(255, 68, 68, 0.8), 0 0 60px rgba(255, 68, 68, 0.4); }
        }
        
        .pnl-percent {
            font-family: 'Orbitron', sans-serif;
            font-size: 1rem;
            margin-top: 5px;
            opacity: 0.8;
        }
        
        /* Confidence Gauge */
        .confidence-gauge {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease, background 0.3s ease;
            position: relative;
        }
        
        .confidence-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: gaugeShine 2s ease-in-out infinite;
        }
        
        @keyframes gaugeShine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        /* Signal Strength Indicator */
        .signal-strength {
            display: flex;
            gap: 4px;
            justify-content: center;
            margin-top: 10px;
        }
        
        .strength-bar {
            width: 8px;
            height: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            transition: all 0.3s ease;
        }
        
        .strength-bar.active {
            animation: strengthPulse 0.8s ease-in-out infinite;
        }
        
        .strength-bar.active.low { background: #ff4444; animation-delay: 0s; }
        .strength-bar.active.medium { background: #ffaa00; animation-delay: 0.1s; }
        .strength-bar.active.high { background: #00ff88; animation-delay: 0.2s; }
        
        @keyframes strengthPulse {
            0%, 100% { opacity: 0.6; transform: scaleY(0.8); }
            50% { opacity: 1; transform: scaleY(1); }
        }
        
        /* Trade Trail Animation */
        .trade-trail {
            position: absolute;
            height: 4px;
            top: 50%;
            transform: translateY(-50%);
            border-radius: 2px;
            transition: all 0.5s ease;
        }
        
        .trade-trail.profit {
            background: linear-gradient(90deg, rgba(0,212,255,0.5), #00ff88);
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        
        .trade-trail.loss {
            background: linear-gradient(90deg, #ff4444, rgba(0,212,255,0.5));
            box-shadow: 0 0 10px rgba(255, 68, 68, 0.5);
        }
        
        /* Position Card Enhanced */
        .position-card-enhanced {
            background: linear-gradient(135deg, rgba(0,0,0,0.6), rgba(0,0,0,0.3));
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(0, 212, 255, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .position-card-enhanced::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 200%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0,212,255,0.1), transparent);
            animation: cardSweep 4s ease-in-out infinite;
        }
        
        @keyframes cardSweep {
            0% { left: -100%; }
            50% { left: 100%; }
            100% { left: 100%; }
        }
        
        /* Position Scale Animations */
        @keyframes scalePulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.95; box-shadow: 0 0 15px rgba(0,255,136,0.8); }
            50% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.7; box-shadow: 0 0 30px rgba(0,255,136,0.5); }
        }
        
        @keyframes ringPulse {
            0% { transform: translate(-50%, -50%) scale(1); opacity: 0.8; }
            100% { transform: translate(-50%, -50%) scale(2.5); opacity: 0; }
        }
        
        @keyframes scaleLabel {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(0.95); }
        }
        
        /* Water Flow Animation - Enhanced with Red/Blue Balance */
        @keyframes waterFlow {
            0% { background-position: 300% 0; }
            100% { background-position: -300% 0; }
        }
        
        @keyframes waterFlowReverse {
            0% { background-position: -300% 0; }
            100% { background-position: 300% 0; }
        }
        
        @keyframes waterRipple {
            0%, 100% { opacity: 0.4; transform: scaleX(1); }
            50% { opacity: 0.8; transform: scaleX(1.02); }
        }
        
        @keyframes waterGlow {
            0%, 100% { box-shadow: 0 0 15px rgba(0,212,255,0.3), inset 0 0 10px rgba(0,212,255,0.1); }
            50% { box-shadow: 0 0 25px rgba(0,212,255,0.5), inset 0 0 15px rgba(0,212,255,0.2); }
        }
        
        @keyframes waterGlowRed {
            0%, 100% { box-shadow: 0 0 15px rgba(255,68,68,0.3), inset 0 0 10px rgba(255,68,68,0.1); }
            50% { box-shadow: 0 0 25px rgba(255,68,68,0.5), inset 0 0 15px rgba(255,68,68,0.2); }
        }
        
        /* Vibrant Position Glow Animations */
        @keyframes borderGlow {
            0%, 100% { opacity: 0.4; filter: hue-rotate(0deg); }
            25% { opacity: 0.7; filter: hue-rotate(30deg); }
            50% { opacity: 0.5; filter: hue-rotate(60deg); }
            75% { opacity: 0.8; filter: hue-rotate(30deg); }
        }
        
        @keyframes profitGlow {
            0%, 100% { text-shadow: 0 0 15px rgba(0,255,136,0.8), 0 0 30px rgba(0,255,136,0.4); }
            50% { text-shadow: 0 0 25px rgba(0,255,136,1), 0 0 50px rgba(0,255,136,0.6), 0 0 80px rgba(0,255,136,0.3); }
        }
        
        @keyframes lossGlow {
            0%, 100% { text-shadow: 0 0 15px rgba(255,68,68,0.8), 0 0 30px rgba(255,68,68,0.4); }
            50% { text-shadow: 0 0 25px rgba(255,68,68,1), 0 0 50px rgba(255,68,68,0.6), 0 0 80px rgba(255,68,68,0.3); }
        }
        
        /* Position control buttons - same pattern as Control Panel */
        .pos-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            font-family: 'Orbitron', sans-serif;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: bold;
        }
        .pos-btn:hover { transform: translateY(-2px); filter: brightness(1.2); }
        .pos-btn:active { transform: translateY(0); }
        .pos-btn-hold {
            background: linear-gradient(135deg, #ffaa00, #ff8800);
            color: #000;
        }
        .pos-btn-hold:hover { box-shadow: 0 5px 20px rgba(255, 170, 0, 0.5); }
        .pos-btn-close {
            background: linear-gradient(135deg, #ff4444, #cc0000);
            color: #fff;
        }
        .pos-btn-close:hover { box-shadow: 0 5px 20px rgba(255, 68, 68, 0.5); }
        
        @keyframes holdPulse {
            0%, 100% { 
                box-shadow: 0 0 15px rgba(0,255,136,0.5), inset 0 0 8px rgba(0,255,136,0.2);
                transform: scale(1);
            }
            50% { 
                box-shadow: 0 0 30px rgba(0,255,136,0.8), inset 0 0 15px rgba(0,255,136,0.4);
                transform: scale(1.02);
            }
        }
        
        .hold-btn:hover {
            filter: brightness(1.2);
            transform: scale(1.02);
        }
        
        .hold-btn.active {
            background: linear-gradient(135deg, rgba(0,255,136,0.6), rgba(0,200,100,0.5)) !important;
            border-color: #00ff88 !important;
            color: #00ff88 !important;
            box-shadow: 0 0 20px rgba(0,255,136,0.6), inset 0 0 10px rgba(0,255,136,0.3);
            animation: holdPulse 1.5s infinite;
        }
        
        .water-flow-bar {
            position: absolute;
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, 
                transparent 0%,
                rgba(255,100,100,0.2) 8%,
                rgba(0,180,255,0.4) 16%,
                rgba(255,120,120,0.3) 24%,
                rgba(0,212,255,0.5) 32%,
                rgba(255,80,80,0.4) 40%,
                rgba(0,200,255,0.6) 48%,
                rgba(255,100,100,0.4) 56%,
                rgba(0,180,255,0.5) 64%,
                rgba(255,120,120,0.3) 72%,
                rgba(0,212,255,0.4) 80%,
                rgba(255,100,100,0.2) 88%,
                transparent 100%);
            background-size: 300% 100%;
            animation: waterFlow 4s linear infinite, waterGlow 2s ease-in-out infinite;
            backdrop-filter: blur(2px);
        }
        
        .water-flow-bar.profit {
            background: linear-gradient(90deg, 
                transparent 0%,
                rgba(0,100,180,0.15) 5%,
                rgba(0,255,136,0.3) 10%,
                rgba(0,180,255,0.4) 18%,
                rgba(0,255,180,0.5) 26%,
                rgba(0,200,255,0.6) 34%,
                rgba(0,255,136,0.5) 42%,
                rgba(0,220,255,0.6) 50%,
                rgba(0,255,180,0.5) 58%,
                rgba(0,180,255,0.5) 66%,
                rgba(0,255,136,0.4) 74%,
                rgba(0,200,255,0.3) 82%,
                rgba(0,100,180,0.15) 90%,
                transparent 100%);
            background-size: 300% 100%;
            animation: waterFlow 3.5s linear infinite, waterGlow 2s ease-in-out infinite;
        }
        
        .water-flow-bar.loss {
            background: linear-gradient(90deg, 
                transparent 0%,
                rgba(180,50,50,0.15) 5%,
                rgba(255,80,80,0.3) 10%,
                rgba(255,100,50,0.4) 18%,
                rgba(255,60,60,0.5) 26%,
                rgba(255,120,80,0.6) 34%,
                rgba(255,50,50,0.5) 42%,
                rgba(255,100,60,0.6) 50%,
                rgba(255,70,70,0.5) 58%,
                rgba(255,120,80,0.5) 66%,
                rgba(255,60,60,0.4) 74%,
                rgba(255,100,50,0.3) 82%,
                rgba(180,50,50,0.15) 90%,
                transparent 100%);
            background-size: 300% 100%;
            animation: waterFlowReverse 3.5s linear infinite, waterGlowRed 2s ease-in-out infinite;
        }
        
        .position-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .position-side-badge {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.2rem;
            font-weight: bold;
            padding: 8px 20px;
            border-radius: 8px;
        }
        
        .position-side-badge.long {
            background: linear-gradient(135deg, rgba(0,255,136,0.3), rgba(0,255,136,0.1));
            color: #00ff88;
            border: 1px solid rgba(0,255,136,0.5);
            animation: longBadgePulse 2s ease-in-out infinite;
        }
        
        .position-side-badge.short {
            background: linear-gradient(135deg, rgba(255,68,68,0.3), rgba(255,68,68,0.1));
            color: #ff4444;
            border: 1px solid rgba(255,68,68,0.5);
            animation: shortBadgePulse 2s ease-in-out infinite;
        }
        
        @keyframes longBadgePulse {
            0%, 100% { box-shadow: 0 0 10px rgba(0,255,136,0.3); }
            50% { box-shadow: 0 0 25px rgba(0,255,136,0.6); }
        }
        
        @keyframes shortBadgePulse {
            0%, 100% { box-shadow: 0 0 10px rgba(255,68,68,0.3); }
            50% { box-shadow: 0 0 25px rgba(255,68,68,0.6); }
        }
        
        /* Position Slot Styles */
        .position-slot {
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .position-slot:hover {
            box-shadow: 0 6px 25px rgba(0,100,200,0.2);
            border-color: rgba(0,180,255,0.3) !important;
        }
        
        .position-slot.has-position {
            border-color: rgba(0,255,136,0.3) !important;
        }
        
        .position-slot .empty-slot {
            animation: emptySlotPulse 3s ease-in-out infinite;
        }
        
        @keyframes emptySlotPulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        
        .position-symbol {
            font-family: 'Orbitron', sans-serif;
            font-size: 1rem;
            color: #00d4ff;
        }
        
        /* Live ticker animation */
        .price-ticker {
            font-family: 'Orbitron', sans-serif;
            display: inline-block;
            transition: all 0.3s ease;
        }
        
        .price-ticker.tick-up {
            color: #00ff88;
            animation: tickUp 0.3s ease;
        }
        
        .price-ticker.tick-down {
            color: #ff4444;
            animation: tickDown 0.3s ease;
        }
        
        @keyframes tickUp {
            0% { transform: translateY(5px); opacity: 0.5; }
            100% { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes tickDown {
            0% { transform: translateY(-5px); opacity: 0.5; }
            100% { transform: translateY(0); opacity: 1; }
        }
        
        /* ===== END ENHANCED LIVE TRADING VISUALIZATION ===== */
        
        /* Animated Badges */
        .badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            position: relative;
            overflow: hidden;
            animation: badgePulse 3s infinite;
        }
        @keyframes badgePulse {
            0%, 100% { box-shadow: 0 0 5px currentColor; }
            50% { box-shadow: 0 0 15px currentColor; }
        }
        .badge-success { background: linear-gradient(135deg, #00ff88, #00cc6a); color: #000; }
        .badge-danger { background: linear-gradient(135deg, #ff4444, #cc2222); color: #fff; }
        .badge-warning { background: linear-gradient(135deg, #ffaa00, #cc8800); color: #000; }
        .badge-info { background: linear-gradient(135deg, #00d4ff, #0099cc); color: #000; }
        .badge-neutral { background: linear-gradient(135deg, #555, #333); color: #ccc; }
        
        /* 3D Chart Container */
        .chart-container {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(0, 212, 255, 0.15);
            backdrop-filter: blur(20px);
            transform-style: preserve-3d;
            transition: all 0.4s;
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 0 30px rgba(0, 212, 255, 0.03);
        }
        .chart-container:hover {
            transform: translateY(-3px);
            box-shadow: 
                0 15px 40px rgba(0, 0, 0, 0.4),
                0 0 30px rgba(0, 212, 255, 0.15);
        }
        .chart-container h2 {
            font-family: 'Orbitron', sans-serif;
            margin-bottom: 18px;
            color: #00d4ff;
            font-size: 1.05rem;
            text-transform: uppercase;
            letter-spacing: 2.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .stats-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 2px;
            font-size: 0.85rem;
        }
        .stats-table th, .stats-table td {
            padding: 12px 14px;
            text-align: left;
        }
        .stats-table th { 
            color: var(--primary); 
            font-family: 'Orbitron', sans-serif;
            font-weight: normal; 
            font-size: 0.7rem; 
            text-transform: uppercase;
            letter-spacing: 2px;
            border-bottom: 2px solid rgba(0, 212, 255, 0.15);
            padding-bottom: 14px;
            position: sticky;
            top: 0;
            background: rgba(10, 10, 30, 0.95);
            backdrop-filter: blur(8px);
            z-index: 5;
        }
        .stats-table td {
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }
        .trade-row { 
            transition: all 0.3s;
            border-radius: 6px;
        }
        .trade-row:hover { 
            background: rgba(0, 212, 255, 0.08);
        }
        
        .mtf-grid { display: flex; gap: 15px; margin-top: 15px; }
        .mtf-item {
            flex: 1;
            padding: 15px;
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 212, 255, 0.02));
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(0, 212, 255, 0.2);
            transition: all 0.3s;
        }
        .mtf-item:hover {
            transform: translateY(-5px) scale(1.05);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
        }
        .mtf-tf { 
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem; 
            color: #888; 
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .mtf-trend { 
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem; 
            font-weight: bold; 
        }
        .mtf-trend.bullish { color: #00ff88; text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
        .mtf-trend.bearish { color: #ff4444; text-shadow: 0 0 10px rgba(255, 68, 68, 0.5); }
        .mtf-trend.neutral { color: #888; }
        
        .progress-ring {
            width: 60px;
            height: 60px;
            margin: 0 auto;
        }
        
        /* Animated Status Dot */
        .status-dot {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: statusPulse 2s infinite;
        }
        .status-dot.online { 
            background: #00ff88; 
            box-shadow: 0 0 20px #00ff88, 0 0 40px #00ff88; 
        }
        .status-dot.offline { 
            background: #ff4444; 
            box-shadow: 0 0 20px #ff4444;
        }
        .status-dot.paused { 
            background: #ffaa00; 
            box-shadow: 0 0 20px #ffaa00;
        }
        @keyframes statusPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.7; }
        }
        
        .tf-btn {
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            border: 1px solid rgba(0, 212, 255, 0.3);
            color: #888;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s;
        }
        .tf-btn:hover { 
            background: rgba(0,212,255,0.2); 
            color: #fff;
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.3);
        }
        .tf-btn.active { 
            background: linear-gradient(135deg, #00d4ff, #0099cc); 
            color: #000; 
            border-color: #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        
        /* Floating tooltip */
        .chart-tooltip {
            position: absolute;
            display: none;
            background: rgba(10, 10, 30, 0.7);
            border: 1px solid rgba(0, 212, 255, 0.5);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.8rem;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            min-width: 180px;
            backdrop-filter: blur(10px);
            transition: opacity 0.15s ease;
        }
        @keyframes tooltipPulse {
            0%, 100% { box-shadow: 0 10px 40px rgba(0, 212, 255, 0.4), 0 0 60px rgba(0, 212, 255, 0.2); }
            50% { box-shadow: 0 10px 50px rgba(0, 212, 255, 0.6), 0 0 80px rgba(0, 212, 255, 0.3); }
        }
        
        /* Chart crosshair */
        .chart-crosshair-h, .chart-crosshair-v {
            position: absolute;
            pointer-events: none;
            z-index: 100;
        }
        .chart-crosshair-h {
            width: 100%;
            height: 1px;
            background: linear-gradient(90deg, transparent, #00d4ff 20%, #00d4ff 80%, transparent);
            left: 0;
        }
        .chart-crosshair-v {
            width: 1px;
            height: 100%;
            background: linear-gradient(180deg, transparent, #00d4ff 20%, #00d4ff 80%, transparent);
            top: 0;
        }
        .chart-price-label {
            position: absolute;
            right: 0;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            font-weight: bold;
            z-index: 101;
            transform: translateY(-50%);
        }
        .chart-time-label {
            position: absolute;
            bottom: 0;
            background: linear-gradient(180deg, #00d4ff, #00ff88);
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            font-weight: bold;
            z-index: 101;
            transform: translateX(-50%);
        }
        
        /* Live indicator animation */
        .live-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 8px;
            animation: livePulse 1s ease-in-out infinite;
        }
        @keyframes livePulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.7); }
            50% { opacity: 0.6; box-shadow: 0 0 0 8px rgba(0,255,136,0); }
        }
        .chart-tooltip .tt-header {
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff;
            font-weight: bold;
            font-size: 0.9rem;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        .chart-tooltip .tt-row {
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
        }
        .chart-tooltip .tt-label { color: #888; }
        .chart-tooltip .tt-value {
            color: #fff;
            font-family: 'Orbitron', sans-serif;
            font-weight: 500;
        }
        .chart-tooltip .tt-value.up { color: #00ff88; }
        .chart-tooltip .tt-value.down { color: #ff4444; }
        .chart-tooltip .tt-change {
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid rgba(255,255,255,0.1);
            text-align: center;
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            font-weight: bold;
        }
        
        /* Floating Refresh Button - Draggable */
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            color: #000;
            border: none;
            border-radius: 50%;
            cursor: grab;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 30px rgba(0, 212, 255, 0.5);
            transition: box-shadow 0.3s, transform 0.2s;
            z-index: 1000;
            user-select: none;
        }
        .refresh-btn:hover { 
            box-shadow: 0 12px 40px rgba(0, 212, 255, 0.7);
            transform: scale(1.1);
        }
        .refresh-btn:active {
            cursor: grabbing;
            transform: scale(0.95);
        }
        
        .last-update { 
            position: fixed;
            bottom: 35px;
            left: 30px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            color: #555;
            letter-spacing: 1px;
            z-index: 100;
            background: rgba(0,0,0,0.6);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.05);
            backdrop-filter: blur(8px);
        }
        
        /* Particle Animation Keyframes */
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        /* Neon Glow Text */
        .neon-text {
            animation: neonFlicker 1.5s infinite alternate;
        }
        @keyframes neonFlicker {
            0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {
                text-shadow: 
                    0 0 10px #00d4ff,
                    0 0 20px #00d4ff,
                    0 0 30px #00d4ff,
                    0 0 40px #00d4ff;
            }
            20%, 24%, 55% {
                text-shadow: none;
            }
        }
        
        /* AI Info Buttons */
        .info-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 212, 255, 0.05));
            border: 2px solid rgba(0, 212, 255, 0.4);
            color: #00d4ff;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-left: 10px;
            transition: all 0.3s ease;
            position: relative;
            z-index: 10;
        }
        .info-btn:hover {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.4), rgba(0, 212, 255, 0.2));
            transform: scale(1.15);
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        }
        .info-btn:active {
            transform: scale(0.95);
        }
        .info-btn.loading {
            animation: infoPulse 1s infinite;
        }
        @keyframes infoPulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }
            50% { opacity: 0.6; box-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
        }
        
        .section-title-wrapper {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 25px 0 15px 0;
        }
        .section-title-wrapper .section-title {
            margin: 0;
            flex: 1;
        }
        
        /* Market Scanner Styles */
        .market-scanner {
            background: linear-gradient(135deg, rgba(10, 15, 30, 0.9), rgba(20, 30, 60, 0.7));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 16px;
            padding: 20px;
            margin: 25px 0;
        }
        .scanner-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }
        .scanner-card {
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.2));
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .scanner-card:hover {
            transform: translateY(-3px);
            border-color: #00d4ff;
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        }
        .scanner-card.has-position {
            border-color: #ff9800;
            box-shadow: 0 0 20px rgba(255, 152, 0, 0.4);
            background: linear-gradient(180deg, rgba(255, 152, 0, 0.15) 0%, rgba(30, 35, 45, 0.98) 100%);
        }
        .scanner-card.has-position::before {
            content: '📍';
            position: absolute;
            top: 5px;
            right: 8px;
            font-size: 14px;
        }
        .scanner-card.active {
            border-color: #00ff88;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
        }
        .scanner-card.active::before {
            content: '✓ PRIMARY';
            position: absolute;
            top: 5px;
            right: 8px;
            color: #00ff88;
            font-size: 10px;
            background: rgba(0, 255, 136, 0.2);
            padding: 2px 5px;
            border-radius: 3px;
        }
        .position-badge {
            display: inline-block;
            background: linear-gradient(135deg, #ff9800, #ff5722);
            color: #fff;
            font-size: 0.6rem;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 6px;
            font-family: 'Rajdhani', sans-serif;
            vertical-align: middle;
            animation: pulse-badge 2s infinite;
        }
        @keyframes pulse-badge {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .scanner-symbol {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            font-weight: bold;
            color: #fff;
            margin-bottom: 8px;
        }
        .scanner-price {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            color: #00d4ff;
            margin-bottom: 5px;
        }
        .scanner-change {
            font-size: 0.85rem;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
        }
        .scanner-change.up {
            color: #00ff88;
            background: rgba(0, 255, 136, 0.15);
        }
        .scanner-change.down {
            color: #ff4444;
            background: rgba(255, 68, 68, 0.15);
        }
        .scanner-stats {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 0.7rem;
            color: #888;
        }
        .scanner-volatility {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .volatility-bar {
            width: 40px;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }
        .volatility-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.5s ease;
        }
        .volatility-fill.low { background: #00ff88; }
        .volatility-fill.medium { background: #ffaa00; }
        .volatility-fill.high { background: #ff4444; }
        
        /* Score badge */
        .scanner-score {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            min-width: 28px;
            text-align: center;
        }
        .scanner-score.high { background: rgba(0, 255, 136, 0.3); color: #00ff88; }
        .scanner-score.medium { background: rgba(255, 170, 0, 0.3); color: #ffaa00; }
        .scanner-score.low { background: rgba(136, 136, 136, 0.3); color: #888; }
        
        /* Signal badges */
        .signal-badge {
            font-size: 0.65rem;
            padding: 1px 4px;
            border-radius: 3px;
            margin-left: 6px;
            font-weight: bold;
            animation: signalPulse 1.5s infinite;
        }
        .signal-badge.long { background: rgba(0, 255, 136, 0.3); color: #00ff88; }
        .signal-badge.short { background: rgba(255, 68, 68, 0.3); color: #ff4444; }
        @keyframes signalPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        /* Indicator chips */
        .scanner-indicators {
            display: flex;
            gap: 4px;
            margin-top: 6px;
            flex-wrap: wrap;
        }
        .ind-chip {
            font-size: 0.6rem;
            padding: 2px 5px;
            border-radius: 3px;
            background: rgba(0, 212, 255, 0.15);
            color: #00d4ff;
        }
        .ind-chip.rsi-overbought { background: rgba(255, 68, 68, 0.2); color: #ff6666; }
        .ind-chip.rsi-oversold { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .ind-chip.rsi-neutral { background: rgba(136, 136, 136, 0.2); color: #aaa; }
        .ind-chip.trend-bullish { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .ind-chip.trend-bearish { background: rgba(255, 68, 68, 0.2); color: #ff6666; }
        
        /* PhD Math Score Chips */
        .scanner-math-scores {
            display: flex;
            gap: 6px;
            margin-top: 6px;
            justify-content: center;
        }
        .math-chip {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.65rem;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            min-width: 32px;
            text-align: center;
        }
        .math-chip.long-chip { border: 1px solid rgba(0, 255, 136, 0.4); }
        .math-chip.short-chip { border: 1px solid rgba(255, 68, 68, 0.4); }
        .math-chip.high { background: rgba(0, 255, 136, 0.25); color: #00ff88; }
        .math-chip.medium { background: rgba(255, 170, 0, 0.25); color: #ffaa00; }
        .math-chip.low { background: rgba(136, 136, 136, 0.2); color: #888; }
        
        /* Rank badge */
        .scanner-rank {
            position: absolute;
            top: 6px;
            left: 8px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.6rem;
            font-weight: bold;
            color: rgba(255, 255, 255, 0.35);
        }
        .scanner-rank.top3 { color: #00d4ff; text-shadow: 0 0 6px rgba(0, 212, 255, 0.5); }
        .scanner-rank.rank1 { color: #ffd700; text-shadow: 0 0 8px rgba(255, 215, 0, 0.6); font-size: 0.7rem; }
        
        /* Volume display */
        .scanner-volume {
            font-size: 0.65rem;
            color: #777;
            margin-top: 2px;
        }
        .scanner-volume .vol-value { color: #aaa; }
        .scanner-volume .vol-ratio {
            padding: 1px 4px;
            border-radius: 3px;
            margin-left: 4px;
            font-weight: bold;
        }
        .scanner-volume .vol-ratio.hot { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .scanner-volume .vol-ratio.normal { background: rgba(136, 136, 136, 0.15); color: #888; }
        .scanner-volume .vol-ratio.dead { background: rgba(255, 68, 68, 0.15); color: #ff6666; }
        
        /* Momentum chip */
        .momentum-chip {
            font-size: 0.6rem;
            font-weight: bold;
            padding: 1px 5px;
            border-radius: 3px;
        }
        .momentum-chip.bullish { background: rgba(0, 255, 136, 0.15); color: #00ff88; }
        .momentum-chip.bearish { background: rgba(255, 68, 68, 0.15); color: #ff4444; }
        .momentum-chip.flat { background: rgba(136, 136, 136, 0.1); color: #666; }
        
        /* Zone warning badge */
        .zone-warning {
            font-size: 0.6rem;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            margin-top: 4px;
            text-align: center;
            animation: zoneFlash 1.5s infinite;
        }
        .zone-warning.resistance { background: rgba(255, 68, 68, 0.25); color: #ff4444; border: 1px solid rgba(255, 68, 68, 0.4); }
        .zone-warning.support { background: rgba(0, 255, 136, 0.2); color: #00ff88; border: 1px solid rgba(0, 255, 136, 0.3); }
        .zone-warning.caution { background: rgba(255, 170, 0, 0.2); color: #ffaa00; border: 1px solid rgba(255, 170, 0, 0.3); }
        @keyframes zoneFlash {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        /* Bottom row with vol + momentum */
        .scanner-bottom-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            gap: 4px;
        }
        
        .scanner-pulse {
            animation: scannerPulse 2s infinite;
        }
        @keyframes scannerPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .scanner-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .scanner-refresh-time {
            font-size: 0.7rem;
            color: #555;
            font-family: 'Orbitron', sans-serif;
        }
        
        .ai-analyze-btn {
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            color: #000;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .ai-analyze-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        }
        .ai-analyze-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .scanner-recommendation {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 212, 255, 0.1));
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 10px;
            padding: 12px 15px;
            margin-top: 15px;
            display: none;
        }
        .scanner-recommendation.show {
            display: block;
            animation: fadeSlideIn 0.3s ease;
        }
        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .recommendation-title {
            font-family: 'Orbitron', sans-serif;
            color: #00ff88;
            font-size: 0.85rem;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .recommendation-text {
            color: #ddd;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        /* AI Info Modal */
        .ai-modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            z-index: 9999;
            animation: modalFadeIn 0.3s ease;
        }
        @keyframes modalFadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .ai-modal {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            background: linear-gradient(135deg, rgba(10, 15, 30, 0.98), rgba(20, 30, 60, 0.95));
            border: 2px solid #00d4ff;
            border-radius: 20px;
            box-shadow: 
                0 20px 60px rgba(0, 212, 255, 0.3),
                0 0 100px rgba(0, 212, 255, 0.1),
                inset 0 0 60px rgba(0, 212, 255, 0.05);
            animation: modalSlideIn 0.3s ease;
            overflow: hidden;
        }
        @keyframes modalSlideIn {
            from { transform: translate(-50%, -50%) scale(0.9); opacity: 0; }
            to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }
        .ai-modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 25px;
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.2), transparent);
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        .ai-modal-title {
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .ai-modal-close {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: rgba(255, 68, 68, 0.2);
            border: 1px solid rgba(255, 68, 68, 0.4);
            color: #ff4444;
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }
        .ai-modal-close:hover {
            background: rgba(255, 68, 68, 0.4);
            transform: scale(1.1);
        }
        .ai-modal-content {
            padding: 25px;
            max-height: 60vh;
            overflow-y: auto;
            color: #ddd;
            font-size: 0.95rem;
            line-height: 1.7;
        }
        .ai-modal-content::-webkit-scrollbar {
            width: 6px;
        }
        .ai-modal-content::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }
        .ai-modal-content::-webkit-scrollbar-thumb {
            background: rgba(0, 212, 255, 0.3);
            border-radius: 3px;
        }
        .ai-modal-loading {
            text-align: center;
            padding: 40px;
            color: #00d4ff;
        }
        .ai-modal-loading .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(0, 212, 255, 0.2);
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .ai-response {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 20px;
            border-left: 3px solid #00d4ff;
        }
        .ai-response h4 {
            color: #00ff88;
            margin-bottom: 10px;
            font-family: 'Orbitron', sans-serif;
        }
        .ai-response p {
            margin-bottom: 12px;
        }
        .ai-response ul {
            margin-left: 20px;
            margin-bottom: 12px;
        }
        .ai-response li {
            margin-bottom: 6px;
        }
        .ai-response code {
            background: rgba(0, 212, 255, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: #00d4ff;
        }
        
        /* ========== ENHANCED UX ADDITIONS ========== */
        
        /* News headline improvements */
        .news-headline-item {
            background: rgba(255,255,255,0.03);
            transition: all 0.25s ease;
            border-radius: 8px;
            padding: 10px 12px !important;
            border-left: 3px solid transparent;
        }
        .news-headline-item:hover {
            background: rgba(255,255,255,0.07);
            border-left-color: #00d4ff;
            transform: translateX(4px);
        }
        .news-headline-link {
            color: #ccc;
            text-decoration: none;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .news-headline-link:hover { color: #00d4ff; }
        
        /* Grid-1 for full width sections */
        .grid-1 {
            margin-bottom: 20px;
        }
        
        /* Improve empty state for tables */
        .stats-table td[colspan] {
            text-align: center;
            padding: 30px 14px;
            font-style: italic;
        }
        
        /* Dashboard footer */
        .dashboard-footer {
            text-align: center;
            padding: 30px 0 60px 0;
            color: #333;
            font-size: 0.7rem;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 2px;
        }
        
        /* Smooth transitions for all section titles */
        .section-title-wrapper {
            transition: opacity 0.3s ease;
        }
        
        /* Improved market scanner card hover */
        .scanner-card {
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .scanner-card:hover {
            transform: translateY(-5px) scale(1.02);
        }
        
        /* Better info button styling */
        .info-btn {
            font-family: 'Orbitron', sans-serif;
            font-weight: 900;
            min-width: 32px;
            min-height: 32px;
        }
    </style>
</head>
<body>
    <script>
        // Setup button handlers on page load
        document.addEventListener('DOMContentLoaded', function() {
            // Attach click handlers to HOLD and CLOSE buttons
            function setupButtonHandlers() {
                var holdBtn1 = document.getElementById('hold-btn-1');
                var closeBtn1 = document.getElementById('close-btn-1');
                var holdBtn2 = document.getElementById('hold-btn-2');
                var closeBtn2 = document.getElementById('close-btn-2');
                
                if (holdBtn1) {
                    holdBtn1.onclick = function(e) {
                        e.preventDefault();
                        console.log('HOLD 1 clicked, symbol:', window.pos1Symbol);
                        if (window.pos1Symbol) toggleHold(window.pos1Symbol, 1);
                        else alert('No position to hold');
                    };
                }
                if (closeBtn1) {
                    closeBtn1.onclick = function(e) {
                        e.preventDefault();
                        console.log('CLOSE 1 clicked, symbol:', window.pos1Symbol);
                        if (window.pos1Symbol) closePosition(window.pos1Symbol);
                        else alert('No position to close');
                    };
                }
                if (holdBtn2) {
                    holdBtn2.onclick = function(e) {
                        e.preventDefault();
                        console.log('HOLD 2 clicked, symbol:', window.pos2Symbol);
                        if (window.pos2Symbol) toggleHold(window.pos2Symbol, 2);
                        else alert('No position to hold');
                    };
                }
                if (closeBtn2) {
                    closeBtn2.onclick = function(e) {
                        e.preventDefault();
                        console.log('CLOSE 2 clicked, symbol:', window.pos2Symbol);
                        if (window.pos2Symbol) closePosition(window.pos2Symbol);
                        else alert('No position to close');
                    };
                }
                console.log('Button handlers attached:', !!holdBtn1, !!closeBtn1, !!holdBtn2, !!closeBtn2);
            }
            
            // Setup handlers after short delay to ensure DOM is ready
            setTimeout(setupButtonHandlers, 500);
            // Also setup after each data load
            window.setupButtonHandlers = setupButtonHandlers;
        });
        });
    </script>    
    </div>
    
    <!-- AI Info Modal -->
    <div class="ai-modal-overlay" id="ai-modal-overlay" onclick="closeAiModal(event)">
        <div class="ai-modal" onclick="event.stopPropagation()">
            <div class="ai-modal-header">
                <div class="ai-modal-title">
                    <span>🤖</span>
                    <span id="ai-modal-topic">AI Explanation</span>
                </div>
                <button class="ai-modal-close" onclick="closeAiModal()">×</button>
            </div>
            <div class="ai-modal-content" id="ai-modal-content">
                <div class="ai-modal-loading">
                    <div class="spinner"></div>
                    <div>AI is thinking...</div>
                </div>
            </div>
        </div>
    </div>
    
    <canvas id="bg-canvas"></canvas>
    <div class="main-content" id="main-content">
    <div class="container">
        <h1>🚀 JULABA TRADING SYSTEM</h1>
        
        <!-- Navigation -->
        <div style="text-align: center; margin-bottom: 24px;">
            <a href="/debug" class="nav-link-btn" style="display: inline-flex; align-items: center; gap: 8px; padding: 12px 28px; background: linear-gradient(135deg, rgba(80,80,100,0.8), rgba(60,60,80,0.8)); color: #ccc; font-family: 'Orbitron', sans-serif; font-size: 0.85rem; font-weight: bold; text-decoration: none; border-radius: 10px; margin: 5px; transition: all 0.3s; border: 1px solid rgba(255,255,255,0.1); letter-spacing: 1px;">🔧 DEBUG</a>
        </div>
        <style>
            .nav-link-btn:hover { transform: translateY(-2px); filter: brightness(1.15); box-shadow: 0 6px 25px rgba(0,0,0,0.3) !important; }
            .nav-link-btn:active { transform: translateY(0); }
        </style>
        
        <!-- Status Row -->
        <div class="grid">
            <div class="card">
                <h3>Status</h3>
                <div class="value">
                    <span class="status-dot" id="status-dot"></span>
                    <span id="status-text">Loading...</span>
                </div>
            </div>
            <div class="card">
                <h3>Balance</h3>
                <div class="value" id="balance">$0.00</div>
                <div class="sub" id="balance-change">--</div>
            </div>
            <div class="card" id="uta-margin-card">
                <h3>💳 Available</h3>
                <div class="value" id="uta-available" style="font-size: 1.2em; color: #00ff88;">$0.00</div>
                <div class="sub" id="uta-borrowed" style="color: #ff9800; display: none;">Borrowed: $0</div>
                <div class="sub" id="uta-ltv" style="font-size: 0.7em; color: #888; display: none;">LTV: 0%</div>
            </div>
            <div class="card">
                <h3>Today's P&L</h3>
                <div class="value" id="today-pnl">$0.00</div>
            </div>
            <div class="card">
                <h3>Total P&L</h3>
                <div class="value" id="total-pnl">$0.00</div>
            </div>
            <div class="card">
                <h3>Win Rate</h3>
                <div class="value" id="win-rate">0%</div>
                <div class="sub" id="win-loss">0W / 0L</div>
            </div>
            <div class="card">
                <h3>Trades</h3>
                <div class="value" id="total-trades">0</div>
            </div>
        </div>
        
        <!-- THE MACHINE - Unified Trading System Visualization -->
        <div class="card-lg machine-section" style="margin-top: 15px; margin-bottom: 25px;">
            <div class="section-title-wrapper">
                <h2 class="section-title" style="text-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.4);">⚙️ THE MACHINE</h2>
                <span id="pipeline-overall-status" class="badge" style="margin-left: 10px;">LOADING</span>
                <span id="pipeline-stage-label" style="margin-left: 12px; font-size: 0.7rem; color: #888; font-family: 'Orbitron', sans-serif;">DATA PIPELINE: Bybit → Scan → Math → AI → Trade</span>
                <button class="info-btn" onclick="askAiInfo('system_architecture', 'Trading System Architecture')" title="Ask AI about this">?</button>
            </div>
            
            <!-- Component Status Bar - Mirrors real pipeline stages -->
            <div id="component-status-bar" style="display: flex; gap: 8px; padding: 10px 14px; background: rgba(0,0,0,0.5); border-radius: 8px; margin-bottom: 12px; flex-wrap: wrap; font-size: 0.68rem; font-family: 'Orbitron', sans-serif; border: 1px solid rgba(255,255,255,0.04);">
                <div class="comp-status engine-status" id="comp-engine" title="Trading Engine">⚙️ <span>--</span></div>
                <div class="comp-status-divider">│</div>
                <div class="comp-status" id="comp-market" title="Bybit Exchange">💱 <span>--</span></div>
                <div class="comp-status" id="comp-indicators" title="Indicators">📊 <span>--</span></div>
                <div class="comp-status" id="comp-regime" title="Regime">🎯 <span>--</span></div>
                <div class="comp-status" id="comp-ai" title="AI Filter">🤖 <span>--</span></div>
                <div class="comp-status" id="comp-risk" title="Risk Manager">🛡️ <span>--</span></div>
                <div class="comp-status" id="comp-position" title="Position">💼 <span>--</span></div>
                <div class="comp-status" id="comp-exchange" title="Exchange">💱 <span>--</span></div>
                <div class="comp-status" id="comp-telegram" title="Telegram">📱 <span>--</span></div>
            </div>
            
            <!-- Engine Metrics Row -->
            <div id="engine-status-bar" style="display: flex; justify-content: space-between; padding: 8px 14px; background: rgba(0,0,0,0.35); border-radius: 8px; margin-bottom: 12px; font-size: 0.72rem; font-family: 'Orbitron', sans-serif; border: 1px solid rgba(255,255,255,0.03);">
                <span>CYCLE: <span id="engine-cycle" style="color: #00d4ff;">--</span></span>
                <span>SIGNAL: <span id="engine-signal" style="color: #888;">--</span></span>
                <span>RISK: <span id="engine-risk" style="color: #00ff88;">OK</span></span>
                <span>AI: <span id="engine-ai" style="color: #888;">--</span></span>
                <span>UPT: <span id="engine-uptime" style="color: #666;">--</span></span>
            </div>
            
            <!-- System Health Panel -->
            <div id="error-panel" style="padding: 10px 15px; background: linear-gradient(135deg, rgba(0,255,100,0.1) 0%, rgba(0,100,50,0.2) 100%); border: 1px solid rgba(0,255,100,0.3); border-radius: 8px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span id="error-panel-title" style="color: #00ff88; font-family: 'Orbitron', sans-serif; font-size: 0.8rem; font-weight: bold;">
                        ✅ SYSTEM HEALTH (<span id="error-count">0</span> errors)
                    </span>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <button id="clear-errors-btn" onclick="clearErrors()" style="display: none; padding: 2px 8px; background: rgba(255,100,100,0.2); border: 1px solid rgba(255,100,100,0.4); border-radius: 4px; color: #ff8888; font-size: 0.65rem; cursor: pointer; font-family: 'Orbitron', sans-serif;">CLEAR</button>
                        <span id="engine-status-badge" style="padding: 2px 8px; background: rgba(0,255,100,0.2); border: 1px solid rgba(0,255,100,0.4); border-radius: 4px; color: #00ff88; font-size: 0.7rem;">ENGINE RUNNING</span>
                    </div>
                </div>
                <div id="error-list" style="font-size: 0.75rem; color: #ffaaaa; max-height: 120px; overflow-y: auto; margin-top: 8px; display: none;"></div>
                <div id="last-error-time" style="font-size: 0.65rem; color: #888; margin-top: 5px; text-align: right; display: none;">--</div>
            </div>
            
            <div style="display: flex; gap: 10px;">
                <!-- Main canvas (left) -->
                <div style="position: relative; height: 280px; flex: 1; background: transparent; border-radius: 10px; overflow: hidden;">
                    <canvas id="machine-canvas" style="width: 100%; height: 100%;"></canvas>
                    <!-- Position Active Flow Indicator -->
                    <div id="pipeline-flow-indicator" class="pipeline-flow-indicator">
                        <div id="flow-dot" class="flow-dot"></div>
                        <span id="flow-status-text">SCANNING...</span>
                    </div>
                </div>
                <!-- Live Event Feed (right) -->
                <div id="machine-event-feed" style="width: 220px; height: 280px; background: rgba(10,15,30,0.8); border: 1px solid #333; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column;">
                    <div style="padding: 8px 10px; background: linear-gradient(90deg, rgba(0,212,255,0.2), transparent); border-bottom: 1px solid #333;">
                        <span style="color: #00d4ff; font-size: 0.8rem; font-weight: bold;">📡 LIVE FEED</span>
                        <span id="machine-status-dot" style="float: right; width: 8px; height: 8px; background: #00ff88; border-radius: 50%; animation: pulse 1s infinite;"></span>
                    </div>
                    <div id="machine-events" style="flex: 1; overflow-y: auto; padding: 5px; font-size: 0.7rem; font-family: 'Rajdhani', monospace;"></div>
                </div>
            </div>
            
            <!-- Alerts Section -->
            <div id="pipeline-alerts" style="margin-top: 10px; display: none;">
                <div style="font-size: 0.75rem; color: #ff4444; padding: 8px; background: rgba(255,68,68,0.1); border: 1px solid rgba(255,68,68,0.3); border-radius: 6px;">
                    <span style="font-weight: bold;">⚠️ ALERTS:</span>
                    <div id="pipeline-alert-list" style="margin-top: 5px;"></div>
                </div>
            </div>
        </div>
        
        <!-- Live Positions - Full Width Row -->
        <div class="grid-1" style="margin-bottom: 20px;">
            <!-- Current Position(s) - Enhanced Live View -->
            <div class="card-lg" style="padding: 20px;">
                <div class="section-title-wrapper">
                    <h2 class="section-title">💼 Live Positions</h2>
                    <button class="info-btn" onclick="askAiInfo('current_position', 'Current Position')" title="Ask AI about this">?</button>
                </div>
                <!-- Always show both position slots - ENLARGED -->
                <div id="dual-positions-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                    <!-- Position 1 Slot -->
                    <div id="position-slot-1" class="position-slot" style="min-height: 240px; background: linear-gradient(135deg, rgba(10,15,30,0.7), rgba(20,30,50,0.6), rgba(15,20,40,0.7)); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 16px; border: 2px solid rgba(0,200,255,0.4); padding: 20px; position: relative; box-shadow: 0 8px 32px rgba(0,150,255,0.15), 0 0 60px rgba(0,200,255,0.08), inset 0 1px 0 rgba(255,255,255,0.1); overflow: visible;">
                        <!-- Animated glow border -->
                        <div style="position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px; border-radius: 18px; background: linear-gradient(45deg, rgba(0,255,255,0.3), rgba(255,0,255,0.2), rgba(0,255,136,0.3), rgba(255,170,0,0.2)); z-index: -1; animation: borderGlow 4s ease-in-out infinite; opacity: 0.6; pointer-events: none;"></div>
                        <div class="slot-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; background: rgba(0,0,0,0.6); padding: 12px 15px; border-radius: 10px; margin: -20px -20px 14px -20px; position: relative; z-index: 100;">
                            <span style="color: #00ffff; font-size: 14px; font-weight: bold; text-transform: uppercase; letter-spacing: 3px; text-shadow: 0 0 15px rgba(0,255,255,0.8), 0 0 30px rgba(0,255,255,0.4);">⚡ POSITION 1</span>
                            <div id="pos1-buttons" style="display: flex; gap: 8px; position: relative; z-index: 101;">
                                <button id="hold-btn-1" class="pos-btn pos-btn-hold" onclick="if(window.toggleHold) window.toggleHold(window.pos1Symbol, 1); else alert('Loading...');" style="position: relative; z-index: 102;">🔒 HOLD</button>
                                <button id="close-btn-1" class="pos-btn pos-btn-close" onclick="if(window.closePosition) window.closePosition(window.pos1Symbol); else alert('Loading...');" style="position: relative; z-index: 102;">✕ CLOSE</button>
                            </div>
                        </div>
                        <div id="pos1-content">
                            <!-- Position 1 content rendered by JS -->
                            <div class="empty-slot" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 160px; color: #555; text-align: center;">
                                <div style="font-size: 42px; margin-bottom: 12px; opacity: 0.3; filter: grayscale(0.5);">📊</div>
                                <div style="font-size: 15px; text-transform: uppercase; letter-spacing: 3px; font-family: 'Orbitron', sans-serif; color: #556;">Waiting for Trade</div>
                                <div style="font-size: 12px; color: #445; margin-top: 10px; display: flex; align-items: center; gap: 6px;"><span class="live-dot" style="width: 6px; height: 6px;"></span> AI Scanning Markets</div>
                            </div>
                        </div>
                    </div>
                    <!-- Position 2 Slot -->
                    <div id="position-slot-2" class="position-slot" style="min-height: 240px; background: linear-gradient(135deg, rgba(10,15,30,0.7), rgba(20,30,50,0.6), rgba(15,20,40,0.7)); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 16px; border: 2px solid rgba(180,0,255,0.4); padding: 20px; position: relative; box-shadow: 0 8px 32px rgba(150,0,255,0.15), 0 0 60px rgba(180,0,255,0.08), inset 0 1px 0 rgba(255,255,255,0.1); overflow: visible;">
                        <!-- Animated glow border -->
                        <div style="position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px; border-radius: 18px; background: linear-gradient(45deg, rgba(180,0,255,0.3), rgba(255,100,200,0.2), rgba(0,255,200,0.3), rgba(255,200,0,0.2)); z-index: -1; animation: borderGlow 4s ease-in-out infinite reverse; opacity: 0.6; pointer-events: none;"></div>
                        <div class="slot-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; background: rgba(0,0,0,0.6); padding: 12px 15px; border-radius: 10px; margin: -20px -20px 14px -20px; position: relative; z-index: 100;">
                            <span style="color: #dd88ff; font-size: 14px; font-weight: bold; text-transform: uppercase; letter-spacing: 3px; text-shadow: 0 0 15px rgba(200,100,255,0.8), 0 0 30px rgba(180,0,255,0.4);">⚡ POSITION 2</span>
                            <div id="pos2-buttons" style="display: flex; gap: 8px; position: relative; z-index: 101;">
                                <button id="hold-btn-2" class="pos-btn pos-btn-hold" onclick="if(window.toggleHold) window.toggleHold(window.pos2Symbol, 2); else alert('Loading...');" style="position: relative; z-index: 102;">🔒 HOLD</button>
                                <button id="close-btn-2" class="pos-btn pos-btn-close" onclick="if(window.closePosition) window.closePosition(window.pos2Symbol); else alert('Loading...');" style="position: relative; z-index: 102;">✕ CLOSE</button>
                            </div>
                        </div>
                        <div id="pos2-content">
                            <!-- Position 2 content rendered by JS -->
                            <div class="empty-slot" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 160px; color: #555; text-align: center;">
                                <div style="font-size: 42px; margin-bottom: 12px; opacity: 0.3; filter: grayscale(0.5);">📊</div>
                                <div style="font-size: 15px; text-transform: uppercase; letter-spacing: 3px; font-family: 'Orbitron', sans-serif; color: #556;">Waiting for Trade</div>
                                <div style="font-size: 12px; color: #445; margin-top: 10px; display: flex; align-items: center; gap: 6px;"><span class="live-dot" style="width: 6px; height: 6px;"></span> AI Scanning Markets</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- LIVE NEWS & MARKET SENTIMENT - Full Width Row -->
        <div class="grid-1" style="margin-bottom: 20px;">
            <div class="card-lg" style="padding: 22px; background: linear-gradient(135deg, rgba(10,15,30,0.8), rgba(20,25,45,0.7)); border: 1px solid rgba(255,170,0,0.25);">
                <div class="section-title-wrapper">
                    <h2 class="section-title">📰 Live Market News & Sentiment</h2>
                    <button class="info-btn" onclick="refreshNews()" title="Refresh News" style="font-size: 14px;">🔄</button>
                </div>
                <div id="news-container" style="margin-top: 15px;">
                    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px;">
                        <!-- Sentiment Panel -->
                        <div id="sentiment-panel" style="background: rgba(0,0,0,0.3); border-radius: 12px; padding: 15px; border: 1px solid rgba(255,255,255,0.1);">
                            <div style="text-align: center; margin-bottom: 15px;">
                                <div style="font-size: 11px; color: #999; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 10px; font-family: 'Orbitron', sans-serif;">Fear & Greed Index</div>
                                <div id="fear-greed-value" style="font-size: 52px; font-weight: bold; color: #ffaa00; text-shadow: 0 0 25px rgba(255,170,0,0.5); font-family: 'Orbitron', sans-serif; line-height: 1;">--</div>
                                <div id="fear-greed-label" style="font-size: 13px; color: #aaa; text-transform: uppercase; letter-spacing: 2px; margin-top: 6px;">Loading...</div>
                            </div>
                            <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="color: #888; font-size: 12px;">24h Market Change</span>
                                    <span id="market-cap-change" style="font-size: 14px; font-weight: bold;">--</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="color: #888; font-size: 12px;">BTC Dominance</span>
                                    <span id="btc-dominance" style="color: #f7931a; font-size: 14px;">--</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #888; font-size: 12px;">News Sentiment</span>
                                    <span id="news-sentiment" style="font-size: 14px;">--</span>
                                </div>
                            </div>
                            <!-- Whale Alerts Summary -->
                            <div id="whale-alerts-panel" style="margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px;">
                                <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">🐋 Whale Activity</div>
                                <div id="whale-alerts-list" style="display: flex; flex-direction: column; gap: 4px; max-height: 80px; overflow-y: auto;">
                                    <div style="color: #555; font-size: 11px;">No whale alerts</div>
                                </div>
                            </div>
                            <div id="sentiment-recommendation" style="margin-top: 12px; padding: 10px; background: rgba(0,255,136,0.1); border-radius: 8px; border: 1px solid rgba(0,255,136,0.3); font-size: 12px; color: #00ff88; text-align: center;">
                                Loading recommendation...
                            </div>
                        </div>
                        <!-- News Headlines -->
                        <div id="news-headlines" style="background: rgba(0,0,0,0.3); border-radius: 12px; padding: 15px; border: 1px solid rgba(255,255,255,0.1); max-height: 380px; overflow-y: auto;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <div style="font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 2px;">📈 Latest Headlines</div>
                                <div id="news-source-count" style="font-size: 11px; color: #555;">--</div>
                            </div>
                            <div id="headlines-list" style="display: flex; flex-direction: column; gap: 8px;">
                                <div style="color: #666; text-align: center; padding: 20px;">Loading news...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- ML Model Status -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">🧠 ML Model</h2>
                    <button class="info-btn" onclick="askAiInfo('ml_model', 'Machine Learning Model')" title="Ask AI about this">?</button>
                </div>
                <div id="ml-display">
                    <div style="text-align: center; padding: 10px 0;">
                        <span class="badge" id="ml-status-badge" style="font-size: 0.9rem; padding: 8px 18px;">LOADING</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Status</span>
                        <span class="indicator-value info" id="ml-status">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Accuracy</span>
                        <span class="indicator-value" id="ml-accuracy">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Training Samples</span>
                        <span class="indicator-value" id="ml-samples">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Features</span>
                        <span class="indicator-value info" id="ml-features">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Last Prediction</span>
                        <span class="indicator-value" id="ml-last-pred">--</span>
                    </div>
                </div>
            </div>
            
            <!-- AI Decision Tracker -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">🎯 AI Accuracy Tracker</h2>
                    <button class="info-btn" onclick="askAiInfo('ai_tracker', 'AI Decision Tracker')" title="Ask AI about this">?</button>
                </div>
                <div id="ai-tracker-display">
                    <div style="text-align: center; padding: 10px 0;">
                        <span class="badge badge-info" id="ai-tracker-badge" style="font-size: 0.9rem; padding: 8px 18px;">TRACKING</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Total Decisions</span>
                        <span class="indicator-value" id="tracker-total">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Approval Rate</span>
                        <span class="indicator-value info" id="tracker-approval-rate">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Approval Accuracy</span>
                        <span class="indicator-value" id="tracker-approval-accuracy">N/A</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Net AI Value</span>
                        <span class="indicator-value" id="tracker-net-value">$0.00</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Pre-Filter Statistics -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">🛡️ Pre-Filter Stats</h2>
                    <button class="info-btn" onclick="askAiInfo('prefilter', 'Pre-Filter System')" title="Ask AI about this">?</button>
                </div>
                <div id="prefilter-display">
                    <div style="text-align: center; padding: 8px 0;">
                        <span class="badge" id="prefilter-badge" style="font-size: 0.9rem; padding: 6px 14px;">--% PASS</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Total Signals</span>
                        <span class="indicator-value" id="prefilter-total">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Passed → AI</span>
                        <span class="indicator-value positive" id="prefilter-passed">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (Score)</span>
                        <span class="indicator-value negative" id="prefilter-score">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (ADX Low)</span>
                        <span class="indicator-value negative" id="prefilter-adx-low">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (ADX Danger)</span>
                        <span class="indicator-value negative" id="prefilter-adx-danger">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (Volume)</span>
                        <span class="indicator-value negative" id="prefilter-volume">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (Confluence)</span>
                        <span class="indicator-value negative" id="prefilter-confluence">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (BTC Filter)</span>
                        <span class="indicator-value negative" id="prefilter-btc">0</span>
                    </div>
                </div>
            </div>
            
            <!-- Trading Parameters -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">⚙️ Trading Parameters</h2>
                    <button class="info-btn" onclick="askAiInfo('trading_parameters', 'Trading Parameters')" title="Ask AI about this">?</button>
                </div>
                <div id="params-display">
                    <div class="indicator-row">
                        <span class="indicator-name">Risk Per Trade</span>
                        <span class="indicator-value" id="param-risk">2%</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">ATR Multiplier</span>
                        <span class="indicator-value" id="param-atr">1.5x</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">TP1 / TP2 / TP3</span>
                        <span class="indicator-value" id="param-tp">1.5R / 2.5R / 4R</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Position Sizing</span>
                        <span class="indicator-value" id="param-sizing">40% / 35% / 25%</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Trail Trigger</span>
                        <span class="indicator-value" id="param-trail">1.5R</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Market Scanner Section -->
        <div class="market-scanner">
            <div class="scanner-header">
                <div class="section-title-wrapper" style="margin: 0; flex: 1;">
                    <h2 class="section-title">🔍 Top 10 Opportunities</h2>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span id="scan-count-badge" style="background: linear-gradient(135deg, #00d4ff, #9966ff); color: #000; padding: 3px 8px; border-radius: 5px; font-size: 0.65rem; font-weight: bold; font-family: 'Orbitron', sans-serif;">SCANNING 50 PAIRS</span>
                        <span class="scanner-refresh-time" id="scanner-time">Updated: --</span>
                        <button class="ai-analyze-btn" id="ai-analyze-btn" onclick="aiAnalyzeMarkets()">
                            🤖 AI Analyze
                        </button>
                        <button class="info-btn" onclick="askAiInfo('market_scanner', 'Market Scanner')" title="Ask AI about this">?</button>
                    </div>
                </div>
            </div>
            <div class="scanner-recommendation" id="scanner-recommendation">
                <div class="recommendation-title">
                    <span>🎯</span>
                    <span>AI Recommendation</span>
                </div>
                <div class="recommendation-text" id="recommendation-text">
                    Loading AI analysis...
                </div>
            </div>
            <div class="scanner-grid" id="scanner-grid">
                <!-- Scanner cards will be populated by JavaScript -->
                <div class="scanner-card scanner-pulse">
                    <div class="scanner-symbol">Loading...</div>
                    <div class="scanner-price">$--</div>
                </div>
            </div>
        </div>
        
        <!-- Live Price Chart -->
        <div class="chart-container">
            <div class="section-title-wrapper" style="margin: 0 0 10px 0;">
                <h2 style="margin: 0;">📊 <span id="chart-symbol" style="color: #00d4ff;">--/USDT</span> <span class="live-dot"></span></h2>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span id="price-display" style="font-size: 1.4rem; color: #fff; font-family: 'Orbitron', sans-serif; animation: priceGlow 2s ease-in-out infinite; transition: transform 0.2s ease, color 0.3s ease;">$--</span>
                    <button class="info-btn" onclick="askAiInfo('live_price_chart', 'Live Price Chart')" title="Ask AI about this">?</button>
                </div>
            </div>
            <style>
                @keyframes priceGlow {
                    0%, 100% { text-shadow: 0 0 10px rgba(0,212,255,0.5); }
                    50% { text-shadow: 0 0 20px rgba(0,212,255,0.8), 0 0 30px rgba(0,255,136,0.5); }
                }
                #chart-wrapper { position: relative; overflow: hidden; border-radius: 8px; }
            </style>
            <!-- Simple chart controls: 24H default, scroll back up to 30 days -->
            <div style="display: flex; gap: 10px; margin-bottom: 10px; align-items: center;">
                <div style="display: flex; gap: 5px;">
                    <button class="range-btn" data-range="6h" onclick="setDateRange('6h')">6H</button>
                    <button class="range-btn" data-range="24h" onclick="setDateRange('24h')">24H</button>
                    <button class="range-btn active" data-range="7d" onclick="setDateRange('7d')">7D</button>
                    <button class="range-btn" data-range="30d" onclick="setDateRange('30d')">30D</button>
                </div>
                <span style="color: #666;">|</span>
                <input type="date" id="chart-date-picker" class="date-picker" onchange="goToDate()" style="width: 140px;">
                <span style="color: #888; font-size: 0.75rem;">← Pick date</span>
            </div>
            <style>
                .range-btn { 
                    background: rgba(30,30,50,0.8); 
                    border: 1px solid #444; 
                    color: #888; 
                    padding: 6px 14px; 
                    border-radius: 4px; 
                    cursor: pointer; 
                    font-size: 0.8rem;
                    transition: all 0.2s;
                }
                .range-btn:hover { border-color: #ff6600; color: #ff6600; }
                .range-btn.active { background: rgba(255,102,0,0.3); border-color: #ff6600; color: #ff6600; }
                .date-picker {
                    background: rgba(30,30,50,0.8);
                    border: 1px solid #444;
                    color: #00d4ff;
                    padding: 6px 10px;
                    border-radius: 4px;
                    font-size: 0.75rem;
                    font-family: 'Orbitron', monospace;
                }
                .date-picker:focus { border-color: #00d4ff; outline: none; }
            </style>
            <div id="chart-wrapper" style="position: relative; height: 220px;">
                <canvas id="priceChart" style="width: 100%; height: 100%;"></canvas>
                <!-- Crosshair lines -->
                <div id="crosshair-h" class="chart-crosshair-h" style="display: none;"></div>
                <div id="crosshair-v" class="chart-crosshair-v" style="display: none;"></div>
                <div id="price-label" class="chart-price-label" style="display: none;">$--</div>
                <div id="time-label" class="chart-time-label" style="display: none;">--:--</div>
                <!-- Floating tooltip -->
                <div id="chart-float-tooltip" class="chart-tooltip">
                    <div class="tt-row"><span class="tt-label">Time:</span><span class="tt-value" id="tt-time">--</span></div>
                    <div class="tt-row"><span class="tt-label">O:</span><span class="tt-value" id="tt-open">--</span></div>
                    <div class="tt-row"><span class="tt-label">H:</span><span class="tt-value up" id="tt-high">--</span></div>
                    <div class="tt-row"><span class="tt-label">L:</span><span class="tt-value down" id="tt-low">--</span></div>
                    <div class="tt-row"><span class="tt-label">C:</span><span class="tt-value" id="tt-close">--</span></div>
                    <div class="tt-change" id="tt-change">--</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.75rem; color: #666;">
                <span>High: <span id="chart-high" style="color: #00ff88;">--</span></span>
                <span>Low: <span id="chart-low" style="color: #ff4444;">--</span></span>
                <span>Volume: <span id="chart-volume" style="color: #00d4ff;">--</span></span>
            </div>
        </div>
        
        <!-- Position 2 Live Price Chart (only shown when Position 2 is active) -->
        <div id="pos2-chart-container" class="chart-container" style="display: none;">
            <div class="section-title-wrapper" style="margin: 0 0 10px 0;">
                <h2 style="margin: 0;">📊 [P2] <span id="pos2-chart-symbol" style="color: #b366ff;">--/USDT</span> <span class="live-dot" style="background: #b366ff;"></span></h2>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span id="pos2-price-display" style="font-size: 1.2rem; color: #b366ff; font-family: 'Orbitron', sans-serif;">$--</span>
                    <span id="pos2-pnl-display" style="font-size: 1rem; font-family: 'Orbitron', sans-serif;">--</span>
                </div>
            </div>
            <div id="pos2-chart-wrapper" style="position: relative; height: 180px;">
                <canvas id="pos2PriceChart" style="width: 100%; height: 100%;"></canvas>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.75rem; color: #666;">
                <span>Entry: <span id="pos2-chart-entry" style="color: #b366ff;">--</span></span>
                <span>SL: <span id="pos2-chart-sl" style="color: #ff4444;">--</span></span>
                <span>TP1: <span id="pos2-chart-tp1" style="color: #00ff88;">--</span></span>
            </div>
        </div>
        
        <!-- Equity Chart -->
        <div class="chart-container">
            <div class="section-title-wrapper" style="margin: 0 0 10px 0;">
                <h2 style="margin: 0;">📈 Equity Curve</h2>
                <button class="info-btn" onclick="askAiInfo('equity_curve', 'Equity Curve')" title="Ask AI about this">?</button>
            </div>
            <canvas id="equityChart" height="80"></canvas>
        </div>
        
        <!-- Recent Signals -->
        <div class="chart-container">
            <h2>📡 Recent Signals</h2>
            <table class="stats-table" id="signals-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Direction</th>
                        <th>Confidence</th>
                        <th>Entry</th>
                        <th>AI Decision</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="signals-body">
                    <tr><td colspan="6">Loading...</td></tr>
                </tbody>
            </table>
        </div>
        
        <!-- Recent Trades -->
        <div class="chart-container">
            <h2>📊 Recent Trades</h2>
            <table class="stats-table" id="trades-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Pos</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry → Exit</th>
                        <th>Time</th>
                        <th>P&L</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody id="trades-body">
                    <tr><td colspan="8">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    </div>
    
    <!-- SYSTEM LOGS SECTION -->
    <div class="main-content" style="padding-top: 0;">
    <div class="container">
        <div id="system-logs-panel" style="
            background: linear-gradient(145deg, rgba(15, 15, 35, 0.95), rgba(5, 5, 20, 0.98));
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 16px;
            padding: 0;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(0, 212, 255, 0.1);
            overflow: hidden;
            position: relative;
        ">
            <!-- Floating Refresh Button -->
            <button id="logs-refresh-btn" onclick="fetchLogs()" style="
                position: absolute;
                top: 10px;
                right: 10px;
                width: 32px;
                height: 32px;
                background: linear-gradient(135deg, #00d4ff, #00ff88);
                color: #000;
                border: none;
                border-radius: 50%;
                cursor: grab;
                font-size: 1rem;
                font-weight: bold;
                z-index: 100;
                box-shadow: 0 3px 10px rgba(0, 212, 255, 0.5);
                transition: box-shadow 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
            " onmouseover="this.style.boxShadow='0 5px 20px rgba(0, 212, 255, 0.8)';" onmouseout="this.style.boxShadow='0 3px 10px rgba(0, 212, 255, 0.5)';">⟳</button>
            
            <!-- Log Header -->
            <div style="
                background: linear-gradient(90deg, rgba(0, 212, 255, 0.15), rgba(0, 255, 136, 0.08), transparent);
                padding: 14px 25px;
                border-bottom: 1px solid rgba(0, 212, 255, 0.2);
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <h2 style="
                    font-family: 'Orbitron', sans-serif;
                    color: #00d4ff;
                    font-size: 1rem;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 3px;
                    text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                ">📜 SYSTEM LOGS</h2>
                <span style="font-size: 0.7rem; color: #555; font-family: 'Orbitron', sans-serif;">Auto-refresh: 10s</span>
            </div>
            
            <!-- Log Content Area -->
            <div id="logs-container" style="
                height: 400px;
                overflow-y: auto;
                overflow-x: auto;
                padding: 20px 25px;
                background: rgba(0, 0, 0, 0.5);
            ">
                <pre id="logs-content" style="
                    font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
                    font-size: 0.85rem;
                    line-height: 1.8;
                    margin: 0;
                    color: #bbb;
                    white-space: pre-wrap;
                    word-break: break-word;
                    tab-size: 4;
                ">Loading logs...</pre>
            </div>
        </div>
    </div>
    </div>
    
    <!-- Dashboard Footer -->
    <div class="dashboard-footer">
        <div style="margin-bottom: 8px; color: #444;">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
        JULABA AI TRADING SYSTEM &bull; Live Dashboard &bull; Data refreshes every 2s
    </div>
    
    <!-- Draggable Refresh Button Script -->
    <script>
        (function() {
            var btn = document.getElementById('logs-refresh-btn');
            let isDragging = false;
            let startX, startY, startLeft, startTop;
            
            btn.addEventListener('mousedown', function(e) {
                isDragging = true;
                btn.style.cursor = 'grabbing';
                startX = e.clientX;
                startY = e.clientY;
                var rect = btn.getBoundingClientRect();
                var parent = btn.parentElement.getBoundingClientRect();
                startLeft = rect.left - parent.left;
                startTop = rect.top - parent.top;
                e.preventDefault();
            });
            
            document.addEventListener('mousemove', function(e) {
                if (!isDragging) return;
                var dx = e.clientX - startX;
                var dy = e.clientY - startY;
                btn.style.left = (startLeft + dx) + 'px';
                btn.style.top = (startTop + dy) + 'px';
                btn.style.right = 'auto';
            });
            
            document.addEventListener('mouseup', function() {
                if (isDragging) {
                    isDragging = false;
                    btn.style.cursor = 'grab';
                }
            });
        })();
    </script>
    
    <div class="last-update">Last update: <span id="last-update">--</span> | Auto-refresh: <span id="auto-refresh-status" style="color:#00ff88;">ON</span></div>
    <button id="main-refresh-btn" class="refresh-btn" onclick="refreshAll()" title="Drag to move, Click to refresh">⚡</button>
    
    <!-- Draggable Main Refresh Button Script -->
    <script>
        (function() {
            var btn = document.getElementById('main-refresh-btn');
            let isDragging = false;
            let hasMoved = false;
            let startX, startY, startLeft, startBottom, startRight;
            
            btn.addEventListener('mousedown', function(e) {
                isDragging = true;
                hasMoved = false;
                btn.style.cursor = 'grabbing';
                startX = e.clientX;
                startY = e.clientY;
                var rect = btn.getBoundingClientRect();
                startRight = window.innerWidth - rect.right;
                startBottom = window.innerHeight - rect.bottom;
                e.preventDefault();
            });
            
            document.addEventListener('mousemove', function(e) {
                if (!isDragging) return;
                var dx = e.clientX - startX;
                var dy = e.clientY - startY;
                if (Math.abs(dx) > 3 || Math.abs(dy) > 3) hasMoved = true;
                btn.style.right = Math.max(10, startRight - dx) + 'px';
                btn.style.bottom = Math.max(10, startBottom + dy) + 'px';
            });
            
            document.addEventListener('mouseup', function() {
                if (isDragging) {
                    isDragging = false;
                    btn.style.cursor = 'grab';
                }
            });
            
            // Prevent click from firing after drag
            btn.addEventListener('click', function(e) {
                if (hasMoved) {
                    e.stopPropagation();
                    e.preventDefault();
                }
            }, true);
        })();
    </script>
    
    <!-- 3D Animated Background -->
    <script>
        // Three.js Particle Background - wrapped in try-catch for safety
        try {
        var canvas = document.getElementById('bg-canvas');
        if (typeof THREE === 'undefined') throw new Error('THREE.js not loaded');
        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        var renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        
        // Create particles
        var particlesGeometry = new THREE.BufferGeometry();
        var particleCount = 2000;
        var posArray = new Float32Array(particleCount * 3);
        var colorsArray = new Float32Array(particleCount * 3);
        
        for(let i = 0; i < particleCount * 3; i += 3) {
            posArray[i] = (Math.random() - 0.5) * 50;
            posArray[i + 1] = (Math.random() - 0.5) * 50;
            posArray[i + 2] = (Math.random() - 0.5) * 50;
            
            // Cyan to green gradient colors
            var t = Math.random();
            colorsArray[i] = t * 0 + (1-t) * 0;        // R
            colorsArray[i + 1] = t * 1 + (1-t) * 0.83; // G
            colorsArray[i + 2] = t * 0.53 + (1-t) * 1; // B
        }
        
        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
        
        var particlesMaterial = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });
        
        var particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
        scene.add(particlesMesh);
        
        // Add connecting lines
        var linesGeometry = new THREE.BufferGeometry();
        var linePositions = [];
        for(let i = 0; i < 100; i++) {
            var x1 = (Math.random() - 0.5) * 30;
            var y1 = (Math.random() - 0.5) * 30;
            var z1 = (Math.random() - 0.5) * 30;
            var x2 = x1 + (Math.random() - 0.5) * 5;
            var y2 = y1 + (Math.random() - 0.5) * 5;
            var z2 = z1 + (Math.random() - 0.5) * 5;
            linePositions.push(x1, y1, z1, x2, y2, z2);
        }
        linesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        var linesMaterial = new THREE.LineBasicMaterial({ 
            color: 0x00d4ff, 
            transparent: true, 
            opacity: 0.15 
        });
        var lines = new THREE.LineSegments(linesGeometry, linesMaterial);
        scene.add(lines);
        
        camera.position.z = 15;
        
        // Mouse interaction
        let mouseX = 0, mouseY = 0;
        document.addEventListener('mousemove', function(e) {
            mouseX = (e.clientX / window.innerWidth) * 2 - 1;
            mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
        });
        
        // Animation loop
        function animateBg() {
            requestAnimationFrame(animateBg);
            
            particlesMesh.rotation.x += 0.0003;
            particlesMesh.rotation.y += 0.0005;
            lines.rotation.x += 0.0002;
            lines.rotation.y += 0.0003;
            
            // Follow mouse
            particlesMesh.rotation.x += mouseY * 0.0005;
            particlesMesh.rotation.y += mouseX * 0.0005;
            
            renderer.render(scene, camera);
        }
        animateBg();
        
        // Resize handler
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        } catch(e) {
            console.warn('Three.js background failed to load:', e.message);
        }
    </script>
    
    <script>
        console.log('=== JULABA DASHBOARD SCRIPT STARTING ===');
        let equityChart = null;
        let priceChart = null;
        let currentTimeframe = '1m';
        let currentDateRange = '4h';  // 4h, 6h, 24h, 7d - default to 4 hours for visible prediction zone
        let customFromTs = null;      // Custom date range from timestamp
        let customToTs = null;        // Custom date range to timestamp
        let allCandleData = [];       // Store all fetched candles for navigation
        let chartViewStart = 0;       // Current view start index
        let chartViewEnd = 0;         // Current view end index
        let currentVisibleCandles = []; // Currently visible candles for hover
        var autoRefresh = true;
        
        // Toast notification function for main dashboard
        function showToast(message, type) {
            type = type || 'info';
            // Create toast element if it doesn't exist
            var toast = document.getElementById('main-toast');
            if (!toast) {
                toast = document.createElement('div');
                toast.id = 'main-toast';
                toast.style.cssText = 'position: fixed; bottom: 80px; right: 30px; padding: 15px 25px; border-radius: 8px; color: #fff; font-weight: bold; opacity: 0; transform: translateY(20px); transition: all 0.3s; z-index: 10000; font-family: Orbitron, sans-serif;';
                document.body.appendChild(toast);
            }
            
            toast.textContent = message;
            
            if (type === 'success') {
                toast.style.background = 'linear-gradient(135deg, #00ff88, #00cc66)';
                toast.style.color = '#000';
            } else if (type === 'error') {
                toast.style.background = 'linear-gradient(135deg, #ff4444, #cc0000)';
                toast.style.color = '#fff';
            } else {
                toast.style.background = 'linear-gradient(135deg, #00d4ff, #0088cc)';
                toast.style.color = '#000';
            }
            
            toast.style.opacity = '1';
            toast.style.transform = 'translateY(0)';
            
            setTimeout(function() {
                toast.style.opacity = '0';
                toast.style.transform = 'translateY(20px)';
            }, 4000);
        }
        
        // Store last dashboard data for position checks
        window.lastDashboardData = null;
        
        // AI Info Modal Functions
        function askAiInfo(topic, displayName) {
            var modal = document.getElementById('ai-modal-overlay');
            var topicEl = document.getElementById('ai-modal-topic');
            var contentEl = document.getElementById('ai-modal-content');
            
            if (topicEl) topicEl.textContent = displayName;
            if (contentEl) contentEl.innerHTML = '<div class="ai-modal-loading"><div class="spinner"></div><div>AI is analyzing ' + displayName + '...</div></div>';
            if (modal) modal.style.display = 'block';
            
            // Fetch AI explanation
            fetch('/api/ai-explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic: topic, display_name: displayName })
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.error) {
                    contentEl.innerHTML = '<div class="ai-response"><p style="color: #ff4444;">❌ ' + data.error + '</p></div>';
                } else {
                    contentEl.innerHTML = '<div class="ai-response">' + formatAiResponse(data.explanation) + '</div>';
                }
            })
            .catch(function(e) {
                contentEl.innerHTML = '<div class="ai-response"><p style="color: #ff4444;">❌ Error: ' + e.message + '</p></div>';
            });
        }
        
        function formatAiResponse(text) {
            // Simple text formatting - avoid complex regexes that break parsing
            if (!text) return '<p>No response</p>';
            var html = text
                .split('**').map(function(s, i) { return i % 2 ? '<strong>' + s + '</strong>' : s; }).join('')
                .split('`').map(function(s, i) { return i % 2 ? '<code>' + s + '</code>' : s; }).join('');
            html = html.split('\\n\\n').join('</p><p>').split('\\n').join('<br>');
            return '<p>' + html + '</p>';
        }
        
        function closeAiModal(event) {
            if (event && event.target !== document.getElementById('ai-modal-overlay')) {
                return;
            }
            document.getElementById('ai-modal-overlay').style.display = 'none';
        }
        
        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeAiModal();
        });
        
        // ========== LOCK SCREEN LOGIC ==========
        // (All lock screen functions are defined in early script tag after <body>)
        //
        
        // Market Scanner Variables
        var currentSymbol = '';
        var marketScanData = [];
        var symbolsWithPositions = [];  // Track ALL open positions
        
        function fetchMarketScan() {
            fetch('/api/market-scan')
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    if (data.pairs) {
                        marketScanData = data.pairs;
                        window.lastMarketScan = data;  // Store for chart to use #1 opportunity
                        currentSymbol = data.current_symbol || '';
                        symbolsWithPositions = data.symbols_with_positions || [];  // Get actual positions
                        updateMarketScanner(data.pairs, data.current_symbol, symbolsWithPositions);
                    }
                    // Update scan count badge dynamically
                    var scanBadge = document.getElementById('scan-count-badge');
                    if (scanBadge && data.scan_pair_count) {
                        scanBadge.textContent = 'SCANNING ' + data.scan_pair_count + ' PAIRS';
                    }
                    var scanTime = document.getElementById('scanner-time');
                    if (scanTime) scanTime.textContent = 'Updated: ' + new Date().toLocaleTimeString();
                })
                .catch(function(e) { console.error('Market scan error:', e); });
        }
        
        function updateMarketScanner(pairs, activeSymbol, positionSymbols) {
            var grid = document.getElementById('scanner-grid');
            if (!grid || !pairs) return;
            
            // Ensure positionSymbols is an array
            positionSymbols = positionSymbols || [];
            
            // Filter out very volatile pairs (>5% ATR is too risky)
            var MAX_VOLATILITY = 5.0;
            var filteredPairs = pairs.filter(function(p) {
                // Always show pairs we have positions in
                if (positionSymbols.indexOf(p.symbol) >= 0) return true;
                // Filter out high volatility
                return (p.volatility || 0) <= MAX_VOLATILITY;
            });
            
            // Sort pairs: symbols with positions first (sorted by symbol), then by score
            var sortedPairs = filteredPairs.slice().sort(function(a, b) {
                var aHasPos = positionSymbols.indexOf(a.symbol) >= 0;
                var bHasPos = positionSymbols.indexOf(b.symbol) >= 0;
                // Both have positions or neither - sort by score
                if (aHasPos && bHasPos) return (b.score || 0) - (a.score || 0);
                if (aHasPos) return -1;  // a has position, comes first
                if (bHasPos) return 1;   // b has position, comes first
                return (b.score || 0) - (a.score || 0);
            });
            
            // Take only top 10 (plus any positions)
            var TOP_COUNT = 10;
            var displayPairs = sortedPairs.slice(0, TOP_COUNT);
            
            var html = '';
            displayPairs.forEach(function(pair, idx) {
                // Highlight ALL symbols with open positions, not just activeSymbol
                var hasPosition = positionSymbols.indexOf(pair.symbol) >= 0;
                var isActive = pair.symbol === activeSymbol;
                var changeClass = pair.change >= 0 ? 'up' : 'down';
                var changeSign = pair.change >= 0 ? '+' : '';
                var volPct = Math.min(100, (pair.volatility || 0) * 25); // Scale volatility
                var volClass = volPct < 33 ? 'low' : volPct < 66 ? 'medium' : 'high';
                var score = pair.score || 0;
                var scoreClass = score >= 70 ? 'high' : score >= 45 ? 'medium' : 'low';
                
                // PhD Math Scores (new)
                var mathLong = pair.math_long || 0;
                var mathShort = pair.math_short || 0;
                var bestDir = pair.best_direction || 'NONE';
                
                // Signal badge shows PhD math recommended direction
                var signalBadge = '';
                if (bestDir === 'LONG') signalBadge = '<span class="signal-badge long">📊 LONG</span>';
                else if (bestDir === 'SHORT') signalBadge = '<span class="signal-badge short">📊 SHORT</span>';
                else if (pair.signal === 1) signalBadge = '<span class="signal-badge long">LONG</span>';
                else if (pair.signal === -1) signalBadge = '<span class="signal-badge short">SHORT</span>';
                
                // Add position badge if we have a position on this symbol
                var positionBadge = hasPosition ? '<span class="position-badge">📍 POSITION</span>' : '';
                
                // Use 'has-position' class for ALL symbols with positions, 'active' for primary symbol
                var cardClass = 'scanner-card';
                if (hasPosition) cardClass += ' has-position';
                if (isActive) cardClass += ' active';
                
                // Rank badge class
                var rankClass = 'scanner-rank';
                if (idx === 0) rankClass += ' rank1';
                else if (idx < 3) rankClass += ' top3';
                
                // Momentum
                var momentum = pair.momentum || 0;
                var momClass = momentum > 0.3 ? 'bullish' : momentum < -0.3 ? 'bearish' : 'flat';
                var momSign = momentum >= 0 ? '+' : '';
                
                // Volume ratio
                var volRatio = pair.volume_ratio || 1;
                var volRatioClass = volRatio >= 1.5 ? 'hot' : volRatio >= 0.8 ? 'normal' : 'dead';
                var volRatioText = volRatio >= 1.5 ? '🔥' + volRatio.toFixed(1) + 'x' : volRatio.toFixed(1) + 'x';
                
                // Zone warnings
                var zoneHtml = '';
                if (pair.in_resistance_zone) {
                    zoneHtml = '<div class="zone-warning resistance">⚠️ AT RESISTANCE</div>';
                } else if (pair.in_support_zone) {
                    zoneHtml = '<div class="zone-warning support">🛡️ AT SUPPORT</div>';
                } else if (pair.in_resistance_caution) {
                    zoneHtml = '<div class="zone-warning caution">⚡ NEAR RESISTANCE</div>';
                } else if (pair.in_support_caution) {
                    zoneHtml = '<div class="zone-warning caution">⚡ NEAR SUPPORT</div>';
                }
                
                html += '<div class="' + cardClass + '" onclick="switchToSymbol(\\'' + pair.symbol + '\\')" title="#' + (idx+1) + ' | Score: ' + Math.round(score) + ' | Click to switch">';
                
                // Rank number
                html += '<span class="' + rankClass + '">#' + (idx + 1) + '</span>';
                
                // Top row: symbol + score + position badge
                html += '<div class="scanner-header">';
                html += '<div class="scanner-symbol">' + pair.symbol.replace('USDT', '') + positionBadge + '</div>';
                html += '<div class="scanner-score ' + scoreClass + '" title="Combined: Math 60% + Indicators 40%">' + Math.round(score) + '</div>';
                html += '</div>';
                
                // Price and change
                html += '<div class="scanner-price">$' + formatPrice(pair.price) + '</div>';
                var changeVal = pair.change !== undefined && pair.change !== null ? pair.change.toFixed(2) : '--';
                html += '<div class="scanner-change ' + changeClass + '">' + changeSign + changeVal + '%' + signalBadge + '</div>';
                
                // PhD Math scores row
                if (mathLong > 0 || mathShort > 0) {
                    var longClass = mathLong >= 50 ? 'high' : mathLong >= 40 ? 'medium' : 'low';
                    var shortClass = mathShort >= 50 ? 'high' : mathShort >= 40 ? 'medium' : 'low';
                    html += '<div class="scanner-math-scores">';
                    html += '<span class="math-chip long-chip ' + longClass + '" title="PhD Math LONG score">L:' + Math.round(mathLong) + '</span>';
                    html += '<span class="math-chip short-chip ' + shortClass + '" title="PhD Math SHORT score">S:' + Math.round(mathShort) + '</span>';
                    html += '</div>';
                }
                
                // Indicators row: RSI + ADX + Trend + MACD
                if (pair.rsi !== undefined) {
                    var rsiClass = pair.rsi > 70 ? 'overbought' : pair.rsi < 30 ? 'oversold' : 'neutral';
                    html += '<div class="scanner-indicators">';
                    html += '<span class="ind-chip rsi-' + rsiClass + '">RSI ' + Math.round(pair.rsi) + '</span>';
                    html += '<span class="ind-chip">ADX ' + Math.round(pair.adx || 0) + '</span>';
                    html += '<span class="ind-chip trend-' + pair.trend + '">' + (pair.trend === 'bullish' ? '↑' : '↓') + '</span>';
                    html += '<span class="ind-chip ' + (pair.macd_bullish ? 'trend-bullish' : 'trend-bearish') + '">' + (pair.macd_bullish ? 'M↑' : 'M↓') + '</span>';
                    html += '</div>';
                }
                
                // S/R Zone warning (eye-catching)
                if (zoneHtml) html += zoneHtml;
                
                // Bottom row: Volume + Momentum + ATR
                html += '<div class="scanner-bottom-row">';
                html += '<div class="scanner-volume">';
                html += '<span class="vol-value">' + formatVolume(pair.volume) + '</span>';
                html += '<span class="vol-ratio ' + volRatioClass + '">' + volRatioText + '</span>';
                html += '</div>';
                html += '<span class="momentum-chip ' + momClass + '" title="1.5h momentum">' + momSign + momentum.toFixed(1) + '%</span>';
                html += '</div>';
                
                // Volatility bar (ATR%)
                html += '<div class="scanner-stats">';
                html += '<div class="scanner-volatility">';
                html += '<span>ATR:</span>';
                html += '<div class="volatility-bar"><div class="volatility-fill ' + volClass + '" style="width: ' + volPct + '%"></div></div>';
                html += '<span style="font-size:0.55rem;color:#666;">' + (pair.volatility || 0).toFixed(1) + '%</span>';
                html += '</div>';
                html += '</div>';
                html += '</div>';
            });
            
            grid.innerHTML = html;
        }
        
        function formatPrice(price) {
            // Handle undefined/null/NaN
            if (price === undefined || price === null || isNaN(price)) return '--';
            // For trading, we need more precision on all prices
            if (price >= 10000) return price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            if (price >= 100) return price.toFixed(2);
            if (price >= 1) return price.toFixed(4);  // Show 4 decimals for $1-100 (LINK, SOL, etc)
            if (price >= 0.01) return price.toFixed(5);
            return price.toFixed(6);
        }
        
        function formatVolume(vol) {
            // Handle undefined/null/NaN
            if (vol === undefined || vol === null || isNaN(vol)) return '--';
            if (vol >= 1e9) return (vol / 1e9).toFixed(1) + 'B';
            if (vol >= 1e6) return (vol / 1e6).toFixed(1) + 'M';
            if (vol >= 1e3) return (vol / 1e3).toFixed(0) + 'K';
            return vol.toFixed(0);
        }
        
        function switchToSymbol(symbol) {
            if (symbol === currentSymbol) return;
            
            // Check if we have an open position first (from cached data)
            if (window.lastDashboardData && window.lastDashboardData.open_position) {
                showToast('Cannot switch pairs while position is open. Close position first.', 'error');
                return;
            }
            
            
            // Show confirmation
            var confirmSwitch = confirm('Switch trading to ' + symbol + '?');
            if (!confirmSwitch) return;
            
            showToast('Switching to ' + symbol + '...', 'info');
            
            fetch('/api/switch-symbol', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: symbol })
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.success) {
                    currentSymbol = symbol;
                    fetchMarketScan();
                    fetchData();
                    fetchPriceData();
                    showToast('Switched to ' + symbol, 'success');
                } else {
                    showToast('Error: ' + (data.error || 'Failed to switch'), 'error');
                }
            })
            .catch(function(e) { 
                console.error('Switch error:', e);
                showToast('Switch error: ' + e.message, 'error'); 
            });
        }
        
        function aiAnalyzeMarkets() {
            var btn = document.getElementById('ai-analyze-btn');
            var recDiv = document.getElementById('scanner-recommendation');
            var recText = document.getElementById('recommendation-text');
            
            if (btn) btn.disabled = true;
            if (recDiv) recDiv.classList.add('show');
            if (recText) recText.innerHTML = '<span class="scanner-pulse">🤖 AI is analyzing all markets...</span>';
            
            fetch('/api/ai-analyze-markets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (btn) btn.disabled = false;
                if (data.recommendation) {
                    var cacheInfo = '';
                    if (data.cached) {
                        var mins = Math.floor(data.cache_expires_in / 60);
                        var secs = data.cache_expires_in % 60;
                        cacheInfo = '<div style="color:#888;font-size:0.75em;margin-top:8px;">📦 Cached result • refreshes in ' + mins + 'm ' + secs + 's</div>';
                    } else if (data.cache_expires_in) {
                        cacheInfo = '<div style="color:#888;font-size:0.75em;margin-top:8px;">✨ Fresh analysis • valid for 5 minutes</div>';
                    }
                    recText.innerHTML = formatAiResponse(data.recommendation) + cacheInfo;
                } else if (data.error) {
                    recText.innerHTML = '<span style="color:#ff4444;">❌ ' + data.error + '</span>';
                }
            })
            .catch(function(e) {
                if (btn) btn.disabled = false;
                recText.innerHTML = '<span style="color:#ff4444;">❌ Error: ' + e.message + '</span>';
            });
        }
        
        var fetchErrorCount = 0;
        var lastFetchSuccess = false;
        
        // Error tracking
        var lastKnownErrorCount = 0;
        
        function clearErrors() {
            fetch('/api/errors/clear', { method: 'POST' })
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    if (data.success) {
                        fetchErrors();  // Refresh the error display
                        if (typeof addMachineEvent === 'function') {
                            addMachineEvent('🧹', 'Errors cleared', 'success');
                        }
                    }
                })
                .catch(function(e) { console.error('Failed to clear errors:', e); });
        }
        
        function fetchErrors() {
            fetch('/api/errors')
                .then(function(response) { 
                    if (!response.ok) return null;
                    return response.json(); 
                })
                .then(function(data) {
                    if (!data) return;
                    
                    var clearBtn = document.getElementById('clear-errors-btn');
                    var errorPanel = document.getElementById('error-panel');
                    var errorCount = document.getElementById('error-count');
                    var errorList = document.getElementById('error-list');
                    var lastErrorTime = document.getElementById('last-error-time');
                    var panelTitle = document.getElementById('error-panel-title');
                    var engineBadge = document.getElementById('engine-status-badge');
                    
                    // Update engine status badge
                    if (engineBadge) {
                        if (data.engine_running === true) {
                            engineBadge.textContent = 'ENGINE RUNNING';
                            engineBadge.style.background = 'rgba(0,255,100,0.2)';
                            engineBadge.style.borderColor = 'rgba(0,255,100,0.4)';
                            engineBadge.style.color = '#00ff88';
                        } else if (data.cycle_count > 0) {
                            engineBadge.textContent = '⚠️ ENGINE STOPPED';
                            engineBadge.style.background = 'rgba(255,100,0,0.2)';
                            engineBadge.style.borderColor = 'rgba(255,150,0,0.4)';
                            engineBadge.style.color = '#ff9900';
                        } else {
                            engineBadge.textContent = 'STARTING...';
                            engineBadge.style.background = 'rgba(100,100,255,0.2)';
                            engineBadge.style.borderColor = 'rgba(100,100,255,0.4)';
                            engineBadge.style.color = '#8888ff';
                        }
                    }
                    
                    // Update panel based on error count
                    if (data.total_errors > 0) {
                        // Has errors - show red panel and clear button
                        if (clearBtn) clearBtn.style.display = 'block';
                        if (errorPanel) {
                            errorPanel.style.background = 'linear-gradient(135deg, rgba(255,50,50,0.15) 0%, rgba(100,0,0,0.3) 100%)';
                            errorPanel.style.borderColor = 'rgba(255,100,100,0.4)';
                        }
                        if (panelTitle) {
                            panelTitle.innerHTML = '⚠️ SYSTEM ERRORS (<span id="error-count">' + data.total_errors + '</span>)';
                            panelTitle.style.color = '#ff6666';
                        }
                        if (errorList) {
                            errorList.style.display = 'block';
                        }
                        if (lastErrorTime) {
                            lastErrorTime.style.display = 'block';
                        }
                        
                        // Update last error time
                        if (lastErrorTime && data.last_error_time) {
                            var t = new Date(data.last_error_time);
                            lastErrorTime.textContent = 'Last: ' + t.toLocaleTimeString();
                        }
                        
                        // Update error list (show last 5)
                        if (errorList && data.error_history) {
                            var html = '';
                            var recent = data.error_history.slice(-5).reverse();
                            for (var i = 0; i < recent.length; i++) {
                                var e = recent[i];
                                var t = new Date(e.timestamp);
                                var countBadge = (e.count && e.count > 1) ? ' <span style="background: rgba(255,100,100,0.3); padding: 1px 4px; border-radius: 3px; font-size: 0.6rem;">x' + e.count + '</span>' : '';
                                html += '<div style="padding: 4px 0; border-bottom: 1px solid rgba(255,100,100,0.2);">';
                                html += '<span style="color: #888; font-size: 0.65rem;">' + t.toLocaleTimeString() + '</span> ';
                                html += '<span style="color: #ff8888; font-weight: bold;">[' + e.context + ']</span>' + countBadge + ' ';
                                html += '<span style="color: #ffaaaa;">' + e.message.substring(0, 80) + (e.message.length > 80 ? '...' : '') + '</span>';
                                html += '</div>';
                            }
                            errorList.innerHTML = html;
                        }
                        
                        // Flash notification for new errors
                        if (data.total_errors > lastKnownErrorCount && lastKnownErrorCount > 0) {
                            if (typeof addMachineEvent === 'function') {
                                addMachineEvent('⚠️', 'New error: ' + (data.last_error || 'Unknown'), 'error');
                            }
                        }
                        lastKnownErrorCount = data.total_errors;
                    } else {
                        // No errors - show green healthy panel, hide clear button
                        if (clearBtn) clearBtn.style.display = 'none';
                        if (errorPanel) {
                            errorPanel.style.background = 'linear-gradient(135deg, rgba(0,255,100,0.1) 0%, rgba(0,100,50,0.2) 100%)';
                            errorPanel.style.borderColor = 'rgba(0,255,100,0.3)';
                        }
                        if (panelTitle) {
                            panelTitle.innerHTML = '✅ SYSTEM HEALTH (<span id="error-count">0</span> errors)';
                            panelTitle.style.color = '#00ff88';
                        }
                        if (errorList) {
                            errorList.style.display = 'none';
                        }
                        if (lastErrorTime) {
                            lastErrorTime.style.display = 'none';
                        }
                    }
                })
                .catch(function(e) { 
                    console.log('Error fetch failed:', e);
                });
        }
        
        // News and sentiment data
        var lastNewsData = null;
        
        function fetchNews() {
            fetch('/api/news')
                .then(function(response) { 
                    if (!response.ok) throw new Error('HTTP ' + response.status);
                    return response.json(); 
                })
                .then(function(data) {
                    if (data.error) {
                        console.warn('News API error:', data.error);
                        return;
                    }
                    lastNewsData = data;
                    updateNewsDisplay(data);
                })
                .catch(function(e) { 
                    console.warn('News fetch error:', e);
                });
        }
        
        function updateNewsDisplay(data) {
            // Update Fear & Greed
            var fgValue = document.getElementById('fear-greed-value');
            var fgLabel = document.getElementById('fear-greed-label');
            if (data.sentiment && fgValue && fgLabel) {
                var fg = data.sentiment.fear_greed_index || 50;
                fgValue.textContent = fg;
                fgLabel.textContent = data.sentiment.fear_greed_label || 'Unknown';
                
                // Color based on sentiment
                if (fg <= 25) {
                    fgValue.style.color = '#ff4444';
                    fgLabel.style.color = '#ff6666';
                } else if (fg <= 45) {
                    fgValue.style.color = '#ffaa00';
                    fgLabel.style.color = '#ffcc00';
                } else if (fg >= 75) {
                    fgValue.style.color = '#00ff88';
                    fgLabel.style.color = '#88ffaa';
                } else if (fg >= 55) {
                    fgValue.style.color = '#88ff88';
                    fgLabel.style.color = '#aaffaa';
                } else {
                    fgValue.style.color = '#aaaaaa';
                    fgLabel.style.color = '#888888';
                }
            }
            
            // Update market cap change
            var mcChange = document.getElementById('market-cap-change');
            if (data.sentiment && mcChange) {
                var change = data.sentiment.market_cap_change_24h || 0;
                mcChange.textContent = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
                mcChange.style.color = change >= 0 ? '#00ff88' : '#ff4444';
            }
            
            // Update BTC dominance
            var btcDom = document.getElementById('btc-dominance');
            if (data.sentiment && btcDom) {
                btcDom.textContent = (data.sentiment.btc_dominance || 0).toFixed(1) + '%';
            }
            
            // Update news sentiment
            var newsSent = document.getElementById('news-sentiment');
            if (data.news_summary && newsSent) {
                var sentiment = data.news_summary.average_sentiment || 0;
                var bullish = data.news_summary.bullish_count || 0;
                var bearish = data.news_summary.bearish_count || 0;
                newsSent.innerHTML = '<span style="color:#00ff88;">↑' + bullish + '</span> / <span style="color:#ff4444;">↓' + bearish + '</span>';
            }
            
            // Update recommendation
            var recEl = document.getElementById('sentiment-recommendation');
            if (data.recommendation && recEl) {
                var rec = data.recommendation.split('\\n')[0] || 'No data';
                // Color based on F&G status in header
                if (rec.includes('HARD BLOCK') || rec.includes('EXTREME FEAR') || rec.includes('EXTREME GREED')) {
                    recEl.style.background = 'rgba(255,68,68,0.15)';
                    recEl.style.borderColor = 'rgba(255,68,68,0.4)';
                    recEl.style.color = '#ff4444';
                } else if (rec.includes('HIGH FEAR') || rec.includes('FEAR') || rec.includes('GREED')) {
                    recEl.style.background = 'rgba(255,170,0,0.15)';
                    recEl.style.borderColor = 'rgba(255,170,0,0.4)';
                    recEl.style.color = '#ffaa00';
                } else if (rec.includes('NEUTRAL')) {
                    recEl.style.background = 'rgba(0,255,136,0.15)';
                    recEl.style.borderColor = 'rgba(0,255,136,0.4)';
                    recEl.style.color = '#00ff88';
                } else {
                    recEl.style.background = 'rgba(255,170,0,0.15)';
                    recEl.style.borderColor = 'rgba(255,170,0,0.4)';
                    recEl.style.color = '#ffaa00';
                }
                // Show full briefing, not just first line
                var fullRec = data.recommendation.replace(/\\n/g, '<br>');
                fullRec = fullRec.replace(/• /g, '&nbsp;&nbsp;');
                recEl.innerHTML = fullRec;
                recEl.style.textAlign = 'left';
                recEl.style.fontSize = '11px';
                recEl.style.lineHeight = '1.6';
            }
            
            // Update headlines
            var headlinesList = document.getElementById('headlines-list');
            if (data.recent_news && headlinesList) {
                var html = '';
                var news = data.recent_news.slice(0, 10);
                
                // Show source count
                var srcCount = document.getElementById('news-source-count');
                if (srcCount) {
                    var sources = {};
                    news.forEach(function(item) { sources[item.source || 'Unknown'] = true; });
                    srcCount.textContent = news.length + ' articles from ' + Object.keys(sources).length + ' sources';
                }
                
                news.forEach(function(item) {
                    var sentiment = item.sentiment || 0;
                    var icon = sentiment > 0.2 ? '🟢' : sentiment < -0.2 ? '🔴' : '⚪';
                    var priority = item.priority || 'medium';
                    var priorityColor = priority === 'critical' ? '#ff4444' : priority === 'high' ? '#ffaa00' : '#444';
                    
                    // Source color coding
                    var sourceColors = {
                        'CoinTelegraph': '#00a2ff', 'Decrypt': '#00d4aa', 'NewsBTC': '#f7931a',
                        'Bitcoinist': '#ff6600', 'CryptoCompare': '#888888'
                    };
                    var srcColor = sourceColors[item.source] || '#666';
                    
                    // Coin badges
                    var coinBadges = '';
                    if (item.coins && item.coins.length > 0) {
                        item.coins.slice(0, 3).forEach(function(coin) {
                            var coinColor = coin === 'BTC' ? '#f7931a' : coin === 'ETH' ? '#627eea' : coin === 'SOL' ? '#9945ff' : '#00d4ff';
                            coinBadges += '<span style="background:' + coinColor + '22;color:' + coinColor + ';padding:1px 5px;border-radius:3px;font-size:9px;font-weight:bold;margin-left:4px;">' + coin + '</span>';
                        });
                    }
                    
                    // Format time ago
                    var timeAgo = '';
                    if (item.published) {
                        try {
                            var pubDate = new Date(item.published);
                            var now = new Date();
                            var diffMs = now - pubDate;
                            var diffMins = Math.floor(diffMs / 60000);
                            var diffHours = Math.floor(diffMins / 60);
                            var diffDays = Math.floor(diffHours / 24);
                            
                            if (diffMins < 1) timeAgo = 'Just now';
                            else if (diffMins < 60) timeAgo = diffMins + 'm ago';
                            else if (diffHours < 24) timeAgo = diffHours + 'h ago';
                            else if (diffDays < 7) timeAgo = diffDays + 'd ago';
                            else timeAgo = pubDate.toLocaleDateString();
                        } catch(e) { timeAgo = ''; }
                    }
                    
                    var titleText = (item.title || '').substring(0, 100);
                    var linkStart = item.url ? '<a href="' + item.url + '" target="_blank" class="news-headline-link">' : '<span>';
                    var linkEnd = item.url ? '</a>' : '</span>';
                    
                    html += '<div class="news-headline-item" style="padding: 8px 10px; border-radius: 6px; border-left: 3px solid ' + priorityColor + ';">';
                    html += '<div style="font-size: 12px; line-height: 1.4;">' + icon + ' ' + linkStart + titleText + linkEnd + coinBadges + '</div>';
                    html += '<div style="font-size: 10px; color: #666; margin-top: 4px; display: flex; justify-content: space-between; align-items: center;">';
                    html += '<span style="color:' + srcColor + ';font-weight:500;">' + (item.source || 'Unknown') + '</span>';
                    html += '<span style="color: #555;">' + timeAgo + '</span>';
                    html += '</div>';
                    html += '</div>';
                });
                headlinesList.innerHTML = html || '<div style="color: #666; text-align: center; padding: 20px;">No news available</div>';
            }
            
            // Update whale alerts
            var whaleList = document.getElementById('whale-alerts-list');
            if (data.whale_alerts && data.whale_alerts.length > 0 && whaleList) {
                var whaleHtml = '';
                data.whale_alerts.slice(0, 3).forEach(function(whale) {
                    var sig = whale.significance || 'low';
                    var sigColor = sig === 'high' ? '#ff4444' : sig === 'medium' ? '#ffaa00' : '#888';
                    var amount = whale.amount || 0;
                    var coin = whale.coin || '?';
                    var usdVal = whale.usd_value || 0;
                    var usdStr = usdVal >= 1e9 ? (usdVal/1e9).toFixed(1) + 'B' : usdVal >= 1e6 ? (usdVal/1e6).toFixed(0) + 'M' : (usdVal/1e3).toFixed(0) + 'K';
                    whaleHtml += '<div style="display:flex;justify-content:space-between;font-size:11px;padding:2px 0;">';
                    whaleHtml += '<span style="color:' + sigColor + ';">🐋 ' + amount.toLocaleString() + ' ' + coin + '</span>';
                    whaleHtml += '<span style="color:#888;">$' + usdStr + '</span>';
                    whaleHtml += '</div>';
                });
                whaleList.innerHTML = whaleHtml;
            }
        }
        
        function refreshNews() {
            fetchNews();
        }
        
        function fetchData() {
            fetch('/api/data')
                .then(function(response) { 
                    if (!response.ok) {
                        throw new Error('HTTP ' + response.status);
                    }
                    return response.json(); 
                })
                .then(function(data) {
                    window.lastData = data;  // Store globally for pipeline popups
                    updateDashboard(data);
                    var lastUpdateEl = document.getElementById('last-update');
                    if (lastUpdateEl) lastUpdateEl.textContent = new Date().toLocaleTimeString();
                    
                    // Reset error count on success
                    if (!lastFetchSuccess && typeof addMachineEvent === 'function') {
                        addMachineEvent('✅', 'Data connection restored', 'success');
                    }
                    fetchErrorCount = 0;
                    lastFetchSuccess = true;
                })
                .catch(function(e) { 
                    console.error('Fetch error:', e);
                    fetchErrorCount++;
                    lastFetchSuccess = false;
                    
                    // Show error in machine after 2 consecutive failures
                    if (fetchErrorCount >= 2 && typeof addMachineEvent === 'function') {
                        addMachineEvent('❌', 'API Error: ' + e.message, 'error');
                    }
                    
                    // Update machine with no data to show error states
                    if (typeof updateMachineFromData === 'function') {
                        updateMachineFromData(null);
                    }
                });
        }
        
        // === CHART SYMBOL STABILITY: Lock symbol to prevent flickering ===
        // When a position is open: ALWAYS show that position's symbol
        // When no position: Show scanner's #1 pair, but LOCK it so it doesn't
        // flicker every time scanner refreshes with slightly different rankings
        window._chartLockedSymbol = null;      // Currently locked symbol for chart
        window._chartSymbolSource = null;      // 'position' or 'scanner' - why this symbol is shown
        
        function getStableChartSymbol() {
            // PRIORITY 1: Open position — ALWAYS wins, immediately
            if (window.lastDashboardData && window.lastDashboardData.open_position && window.lastDashboardData.open_position.symbol) {
                var posSymbol = window.lastDashboardData.open_position.symbol;
                if (window._chartLockedSymbol !== posSymbol || window._chartSymbolSource !== 'position') {
                    console.log('[Chart Stability] LOCKED to POSITION symbol:', posSymbol);
                    window._chartLockedSymbol = posSymbol;
                    window._chartSymbolSource = 'position';
                    window.currentPos1Symbol = posSymbol;
                }
                return posSymbol;
            }
            
            // PRIORITY 2: Scanner's #1 pair — but only update if meaningfully different
            if (window.lastMarketScan && window.lastMarketScan.pairs && window.lastMarketScan.pairs.length > 0) {
                var scannerTop = window.lastMarketScan.pairs[0].symbol;
                
                // If we were showing a position symbol and position just closed,
                // immediately switch to scanner's top pair
                if (window._chartSymbolSource === 'position') {
                    console.log('[Chart Stability] Position closed → switching to scanner top:', scannerTop);
                    window._chartLockedSymbol = scannerTop;
                    window._chartSymbolSource = 'scanner';
                    window.currentPos1Symbol = scannerTop;
                    return scannerTop;
                }
                
                // If no locked symbol yet, use scanner top
                if (!window._chartLockedSymbol) {
                    console.log('[Chart Stability] Initial lock to scanner top:', scannerTop);
                    window._chartLockedSymbol = scannerTop;
                    window._chartSymbolSource = 'scanner';
                    window.currentPos1Symbol = scannerTop;
                    return scannerTop;
                }
                
                // If scanner top changed, update (scanner legitimately found better pair)
                if (window._chartLockedSymbol !== scannerTop && window._chartSymbolSource === 'scanner') {
                    console.log('[Chart Stability] Scanner top changed:', window._chartLockedSymbol, '->', scannerTop);
                    window._chartLockedSymbol = scannerTop;
                    window.currentPos1Symbol = scannerTop;
                    return scannerTop;
                }
                
                // Otherwise keep the locked symbol stable
                return window._chartLockedSymbol;
            }
            
            // PRIORITY 3: Last known symbol (persists across brief gaps)
            if (window._chartLockedSymbol) {
                return window._chartLockedSymbol;
            }
            if (window.currentPos1Symbol) {
                return window.currentPos1Symbol;
            }
            
            return null;
        }
        
        function fetchPriceData() {
            // If custom timestamps are set, use those instead of range parameter
            if (customFromTs && customToTs) {
                fetchPriceDataWithDates(customFromTs, customToTs);
                return;
            }
            
            // UPPER CHART: Show position symbol when trading, scanner top when idle
            // Use 24h range with 15m candles (~96 candles)
            var range = '24h';
            var timeframe = '15m';
            var symbol = getStableChartSymbol();
            
            if (!symbol) {
                console.log('[Chart] No symbol available yet, waiting...');
                return;
            }
            
            var url = '/api/ohlc?tf=' + timeframe + '&range=' + range + '&symbol=' + symbol;
            console.log('[Position 1 Chart] Fetching', symbol, ':', url);
            
            fetch(url)
                .then(function(response) { return response.json(); })
                .then(function(data) { 
                    console.log('[Position 1 Chart] Received', data.candles ? data.candles.length : 0, 'candles for', symbol);
                    // Store all data for navigation
                    if (data && data.candles) {
                        allCandleData = data.candles;
                        chartViewEnd = allCandleData.length;
                        chartViewStart = 0; // Show all loaded data
                        // Store visible candles for hover (full data when live)
                        currentVisibleCandles = data.candles;
                    }
                    updatePriceChart(data); 
                })
                .catch(function(e) { console.error('[Position 1 Chart] Fetch error:', e); });
        }
        
        function fetchLogs() {
            fetch('/api/logs?count=150')
                .then(function(response) { return response.json(); })
                .then(function(data) { updateLogs(data.logs || []); })
                .catch(function(e) {
                    console.error('Logs fetch error:', e);
                    var logsEl = document.getElementById('logs-content');
                    if (logsEl) logsEl.innerHTML = '<span style="color:#ff4444;">Error loading logs</span>';
                });
        }
        
        function updateLogs(logs) {
            var container = document.getElementById('logs-content');
            if (!logs || logs.length === 0) {
                container.textContent = 'No logs available';
                container.style.color = '#666';
                return;
            }
            
            // Filter out noisy log messages
            var filteredLogs = logs.filter(function(log) {
                var msg = log.message || '';
                if (msg.includes('httpx') || msg.includes('HTTP Request:')) return false;
                if (msg.includes('getUpdates') || msg.includes('telegram.org/bot')) return false;
                if (msg.includes('Signal confirmed with confluence')) return false;
                if (msg.includes('Heartbeat') || msg.includes('heartbeat')) return false;
                return true;
            });
            
            if (filteredLogs.length === 0) {
                container.textContent = 'No significant logs (filtered noise)';
                container.style.color = '#666';
                return;
            }
            
            // Build log entries as formatted text
            var logHtml = '';
            filteredLogs.forEach(function(log) {
                var levelColor = '#888';
                var levelBg = 'rgba(136,136,136,0.2)';
                if (log.level === 'ERROR') { levelColor = '#ff4444'; levelBg = 'rgba(255,68,68,0.3)'; }
                else if (log.level === 'WARNING') { levelColor = '#ffaa00'; levelBg = 'rgba(255,170,0,0.3)'; }
                else if (log.level === 'INFO') { levelColor = '#00d4ff'; levelBg = 'rgba(0,212,255,0.2)'; }
                
                var msgColor = levelColor;
                var msg = log.message || '';
                if (msg.includes('✅')) msgColor = '#00ff88';
                else if (msg.includes('❌') || msg.includes('🚫')) msgColor = '#ff4444';
                else if (msg.includes('🧠')) msgColor = '#9966ff';
                else if (msg.includes('📈') || msg.includes('📊')) msgColor = '#00d4ff';
                
                // Short level indicator
                var levelShort = log.level === 'WARNING' ? 'WARN' : log.level === 'ERROR' ? 'ERR' : 'INF';
                var timeShort = (log.time.split(' ')[1] || log.time).substring(0, 5);
                
                logHtml += '<div style="display:flex; align-items:flex-start; gap:8px; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.03);">' +
                    '<span style="color:#555; font-size:0.7rem; flex-shrink:0;">' + timeShort + '</span>' +
                    '<span style="background:' + levelBg + '; color:' + levelColor + '; padding:1px 5px; border-radius:3px; font-size:0.65rem; flex-shrink:0; font-weight:bold;">' + levelShort + '</span>' +
                    '<span style="color:' + msgColor + '; flex:1; word-break:break-word;">' + msg + '</span>' +
                '</div>';
            });
            
            container.innerHTML = logHtml;
            
            // Scroll to bottom
            var logsContainer = document.getElementById('logs-container');
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
        
        function setDateRange(range) {
            currentDateRange = range;
            document.querySelectorAll('.range-btn').forEach(btn => btn.classList.remove('active'));
            var btn = document.querySelector('.range-btn[data-range="' + range + '"]');
            if (btn) btn.classList.add('active');
            
            // Calculate from/to dates based on range (from X ago to now)
            var now = new Date();
            var from = new Date();
            if (range === '6h') from.setHours(now.getHours() - 6);
            else if (range === '24h') from.setDate(now.getDate() - 1);
            else if (range === '7d') from.setDate(now.getDate() - 7);
            else if (range === '30d') from.setDate(now.getDate() - 30);
            
            // Store custom timestamps
            customFromTs = from.getTime();
            customToTs = now.getTime();
            
            // Clear date picker when using range buttons
            document.getElementById('chart-date-picker').value = '';
            
            fetchPriceDataWithDates(customFromTs, customToTs);
        }
        
        function goToDate() {
            var dateInput = document.getElementById('chart-date-picker').value;
            if (!dateInput) {
                console.log('No date selected');
                return;
            }
            
            console.log('Date picker selected:', dateInput);
            
            // Show 24 hours starting from selected date
            // Create date at midnight UTC to avoid timezone issues
            var from = new Date(dateInput + 'T00:00:00Z');
            var to = new Date(from);
            to.setDate(to.getDate() + 1);
            
            // Store custom timestamps
            customFromTs = from.getTime();
            customToTs = to.getTime();
            
            console.log('Custom date range set:', new Date(customFromTs), 'to', new Date(customToTs));
            
            // Clear active range buttons since using custom date
            document.querySelectorAll('.range-btn').forEach(btn => btn.classList.remove('active'));
            currentDateRange = null;
            
            // Force immediate fetch
            fetchPriceDataWithDates(customFromTs, customToTs);
        }
        
        function fetchPriceDataWithDates(fromTs, toTs) {
            // UPPER CHART: Use stable chart symbol (same logic as fetchPriceData)
            var symbol = getStableChartSymbol() || '';
            var url = '/api/ohlc?tf=' + currentTimeframe + '&from=' + fromTs + '&to=' + toTs + (symbol ? '&symbol=' + symbol : '');
            console.log('Fetching Position 1 chart data from', new Date(fromTs), 'to', new Date(toTs), 'Symbol:', symbol, 'URL:', url);
            fetch(url)
                .then(function(response) { return response.json(); })
                .then(function(data) { 
                    console.log('Received', data.candles ? data.candles.length : 0, 'candles');
                    if (data && data.candles) {
                        allCandleData = data.candles;
                        chartViewEnd = allCandleData.length;
                        chartViewStart = 0; // Show all data
                        currentVisibleCandles = data.candles;
                    }
                    updatePriceChart(data); 
                })
                .catch(function(e) { console.error('Price fetch error:', e); });
        }
        
        function refreshAll() {
            fetchData();
            fetchPriceData();
            fetchLogs();
            fetchMarketScan();
        }
        
        // ===== POSITION 2 CHART =====
        var pos2Chart = null;
        
        function fetchPos2ChartData(symbol, posData) {
            if (!symbol) {
                console.log('[P2 Chart] No symbol provided, skipping fetch');
                return;
            }
            
            // LOWER CHART: Always show Position 2 (additional_positions[0])
            // Fetch 24h of 15m candles for Position 2
            var url = '/api/ohlc?tf=15m&range=24h&symbol=' + symbol;
            console.log('[P2 Chart] Fetching Position 2 (' + symbol + '):', url);
            
            fetch(url)
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    console.log('[P2 Chart] Received', data.candles ? data.candles.length : 0, 'candles for', symbol);
                    if (data && data.candles && data.candles.length > 0) {
                        updatePos2Chart(data.candles, posData);
                    }
                })
                .catch(function(e) { console.error('[P2 Chart] Fetch error:', e); });
        }
        
        function updatePos2Chart(candles, posData) {
            var ctx = document.getElementById('pos2PriceChart');
            if (!ctx) {
                console.error('[P2 Chart] Canvas element not found');
                return;
            }
            
            // Optimization: Skip update if data hasn't changed significantly
            if (window.pos2LastCandles && window.pos2LastCandles.length === candles.length && 
                window.pos2LastCandles[window.pos2LastCandles.length - 1].c === candles[candles.length - 1].c) {
                console.log('[P2 Chart] Data unchanged, skipping render');
                return;
            }
            window.pos2LastCandles = candles;
            
            ctx = ctx.getContext('2d');
            console.log('[P2 Chart] Rendering chart with', candles.length, 'candles');
            
            // Prepare data - candles use short keys: t, o, h, l, c, v
            var labels = [];
            var closeData = [];
            var highData = [];
            var lowData = [];
            
            candles.forEach(function(c) {
                labels.push(new Date(c.t));
                closeData.push({ x: new Date(c.t), y: c.c });
                highData.push({ x: new Date(c.t), y: c.h });
                lowData.push({ x: new Date(c.t), y: c.l });
            });
            
            console.log('[P2 Chart] Processed', closeData.length, 'data points');
            
            // Calculate future space for prediction (30 bars)
            var lastCandle = candles[candles.length - 1];
            var firstCandle = candles[0];
            var lastCandleTime = new Date(lastCandle.t);
            var firstCandleTime = new Date(firstCandle.t);
            var tfIntervalMs = 15 * 60000; // 15m candles
            var futureSpace = 30; // 30 extra candle spaces for future prediction zone
            var chartMaxTime = new Date(lastCandleTime.getTime() + (tfIntervalMs * futureSpace));
            var chartMinTime = firstCandleTime;
            
            // FIXED: Store position data in window scope so plugin can read fresh data on every draw
            window.pos2Data = posData;
            window.pos2Candles = candles;  // Store candles for zone calculation
            
            // Plugin to draw entry/SL/TP lines - reads from window.pos2Data for fresh values
            var pos2LinesPlugin = {
                id: 'pos2Lines',
                beforeDraw: function(chart) {
                    var ctx = chart.ctx;
                    var chartArea = chart.chartArea;
                    var yScale = chart.scales.y;
                    var xScale = chart.scales.x;
                    
                    // Read fresh position data from window scope
                    var posData = window.pos2Data;
                    if (!posData) return;
                    
                    var entryPrice = posData.entry;
                    var slPrice = posData.stop_loss;
                    var tp1Price = posData.tp1;
                    var entryTime = posData.entry_time ? new Date(posData.entry_time) : null;
                    var isLong = posData.side === 'LONG';
                    
                    ctx.save();
                    
                    // Entry line
                    if (entryPrice) {
                        var entryY = yScale.getPixelForValue(entryPrice);
                        if (entryY >= chartArea.top && entryY <= chartArea.bottom) {
                            var entryColor = isLong ? '#b366ff' : '#ff66b3';
                            
                            // Horizontal line
                            ctx.strokeStyle = entryColor;
                            ctx.lineWidth = 2;
                            ctx.setLineDash([]);
                            ctx.beginPath();
                            
                            if (entryTime) {
                                var entryX = xScale.getPixelForValue(entryTime);
                                if (entryX >= chartArea.left) {
                                    // Vertical line at entry
                                    ctx.strokeStyle = entryColor;
                                    ctx.lineWidth = 1;
                                    ctx.setLineDash([3, 3]);
                                    ctx.beginPath();
                                    ctx.moveTo(entryX, chartArea.top);
                                    ctx.lineTo(entryX, chartArea.bottom);
                                    ctx.stroke();
                                    
                                    // Horizontal from entry point
                                    ctx.setLineDash([]);
                                    ctx.lineWidth = 2;
                                    ctx.beginPath();
                                    ctx.moveTo(entryX, entryY);
                                    ctx.lineTo(chartArea.right, entryY);
                                    ctx.stroke();
                                    
                                    // Entry marker dot
                                    ctx.beginPath();
                                    ctx.arc(entryX, entryY, 5, 0, Math.PI * 2);
                                    ctx.fillStyle = entryColor;
                                    ctx.fill();
                                    ctx.strokeStyle = '#fff';
                                    ctx.lineWidth = 1;
                                    ctx.stroke();
                                }
                            } else {
                                ctx.moveTo(chartArea.left, entryY);
                                ctx.lineTo(chartArea.right, entryY);
                                ctx.stroke();
                            }
                            
                            // Label
                            ctx.fillStyle = entryColor;
                            ctx.font = '9px Orbitron';
                            ctx.textAlign = 'right';
                            ctx.fillText('ENTRY', chartArea.right - 5, entryY - 3);
                        }
                    }
                    
                    // SL line
                    if (slPrice) {
                        var slY = yScale.getPixelForValue(slPrice);
                        if (slY >= chartArea.top && slY <= chartArea.bottom) {
                            ctx.strokeStyle = 'rgba(255, 68, 68, 0.7)';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            ctx.beginPath();
                            ctx.moveTo(chartArea.left, slY);
                            ctx.lineTo(chartArea.right, slY);
                            ctx.stroke();
                            
                            ctx.fillStyle = '#ff4444';
                            ctx.font = '8px Orbitron';
                            ctx.textAlign = 'right';
                            ctx.fillText('SL', chartArea.right - 5, slY - 2);
                        }
                    }
                    
                    // TP1 line
                    if (tp1Price) {
                        var tp1Y = yScale.getPixelForValue(tp1Price);
                        if (tp1Y >= chartArea.top && tp1Y <= chartArea.bottom) {
                            ctx.strokeStyle = 'rgba(0, 255, 136, 0.7)';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            ctx.beginPath();
                            ctx.moveTo(chartArea.left, tp1Y);
                            ctx.lineTo(chartArea.right, tp1Y);
                            ctx.stroke();
                            
                            ctx.fillStyle = '#00ff88';
                            ctx.font = '8px Orbitron';
                            ctx.textAlign = 'right';
                            ctx.fillText('TP1', chartArea.right - 5, tp1Y - 2);
                        }
                    }
                    
                    // === RESISTANCE & SUPPORT ZONES ===
                    // Use server-calculated zones if available (from market scan)
                    var candleData = window.pos2Candles;
                    var scanData = window.lastMarketScan;
                    var currentSymbol = window.pos2Data && window.pos2Data.symbol ? window.pos2Data.symbol.replace('/USDT:USDT', 'USDT') : '';
                    
                    // Find zone data from market scan
                    var zoneData = null;
                    if (scanData && scanData.pairs) {
                        for (var i = 0; i < scanData.pairs.length; i++) {
                            if (scanData.pairs[i].symbol === currentSymbol || 
                                scanData.pairs[i].symbol === currentSymbol.replace('USDT', '/USDT:USDT')) {
                                zoneData = scanData.pairs[i];
                                break;
                            }
                        }
                    }
                    
                    // Use server zones if available, otherwise calculate from candles
                    var resistanceUpper, resistanceLower, resistanceShadowLower;
                    var supportUpper, supportLower, supportShadowUpper;
                    var resistanceLevel, resistanceTouches, supportLevel, supportTouches;
                    var zoneWidth, shadowWidth;
                    
                    if (zoneData && zoneData.resistance_upper) {
                        // Use server-calculated zones (bounce-based detection)
                        resistanceUpper = zoneData.resistance_upper;
                        resistanceLower = zoneData.resistance_lower;
                        resistanceShadowLower = zoneData.resistance_caution_lower;
                        resistanceLevel = zoneData.resistance_level || resistanceUpper;
                        resistanceTouches = zoneData.resistance_touches || 0;
                        supportUpper = zoneData.support_upper;
                        supportLower = zoneData.support_lower;
                        supportShadowUpper = zoneData.support_caution_upper;
                        supportLevel = zoneData.support_level || supportLower;
                        supportTouches = zoneData.support_touches || 0;
                        zoneWidth = zoneData.zone_width;
                        shadowWidth = zoneWidth * 1.5;
                    } else if (candleData && candleData.length > 0) {
                        // Fallback: Calculate from candle data
                        var highs = candleData.map(c => c.h);
                        var lows = candleData.map(c => c.l);
                        var high24h = Math.max(...highs);
                        var low24h = Math.min(...lows);
                        
                        // Calculate ATR for zone width
                        var trValues = [];
                        for (var i = 1; i < Math.min(14, candleData.length); i++) {
                            var tr = Math.max(
                                candleData[i].h - candleData[i].l,
                                Math.abs(candleData[i].h - candleData[i-1].c),
                                Math.abs(candleData[i].l - candleData[i-1].c)
                            );
                            trValues.push(tr);
                        }
                        var atr = trValues.length > 0 ? trValues.reduce((a,b) => a+b, 0) / trValues.length : (high24h - low24h) * 0.1;
                        
                        var currentPrice = candleData[candleData.length - 1].c;
                        zoneWidth = atr * 0.5;  // REDUCED from 1.5x to 0.5x ATR
                        var minZone = currentPrice * 0.01;
                        var maxZone = currentPrice * 0.05;
                        zoneWidth = Math.max(minZone, Math.min(maxZone, zoneWidth));
                        shadowWidth = zoneWidth * 1.5;
                        
                        resistanceUpper = high24h;
                        resistanceLower = high24h - zoneWidth;
                        resistanceShadowLower = high24h - zoneWidth - shadowWidth;
                        resistanceLevel = high24h;
                        resistanceTouches = 0;  // Fallback = no bounce data
                        supportUpper = low24h + zoneWidth;
                        supportLower = low24h;
                        supportShadowUpper = low24h + zoneWidth + shadowWidth;
                        supportLevel = low24h;
                        supportTouches = 0;  // Fallback = no bounce data
                    }
                    
                    // Draw zones if we have valid data
                    if (resistanceUpper && supportLower) {
                        // Draw RESISTANCE SHADOW/CAUTION ZONE (orange, very faint)
                        var rShadowUpperY = yScale.getPixelForValue(resistanceLower);
                        var rShadowLowerY = yScale.getPixelForValue(resistanceShadowLower);
                        if (rShadowLowerY >= chartArea.top && rShadowUpperY <= chartArea.bottom) {
                            rShadowUpperY = Math.max(rShadowUpperY, chartArea.top);
                            rShadowLowerY = Math.min(rShadowLowerY, chartArea.bottom);
                            
                            // Gradient from danger zone fading outward
                            var gradient = ctx.createLinearGradient(0, rShadowUpperY, 0, rShadowLowerY);
                            gradient.addColorStop(0, 'rgba(255, 136, 0, 0.06)');
                            gradient.addColorStop(1, 'rgba(255, 136, 0, 0.0)');
                            ctx.fillStyle = gradient;
                            ctx.fillRect(chartArea.left, rShadowUpperY, chartArea.right - chartArea.left, rShadowLowerY - rShadowUpperY);
                            
                            // Dashed line at shadow boundary
                            ctx.strokeStyle = 'rgba(255, 136, 0, 0.25)';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([1, 3]);
                            ctx.beginPath();
                            ctx.moveTo(chartArea.left, rShadowLowerY);
                            ctx.lineTo(chartArea.right, rShadowLowerY);
                            ctx.stroke();
                            ctx.setLineDash([]);
                            
                            // Label
                            ctx.fillStyle = 'rgba(255, 136, 0, 0.5)';
                            ctx.font = '7px Orbitron';
                            ctx.textAlign = 'left';
                            ctx.fillText('⚠️ CAUTION ZONE (reduce size)', chartArea.left + 5, rShadowLowerY - 3);
                        }
                        
                        // Draw RESISTANCE ZONE (red, semi-transparent)
                        var rUpperY = yScale.getPixelForValue(resistanceUpper);
                        var rLowerY = yScale.getPixelForValue(resistanceLower);
                        if (rLowerY >= chartArea.top && rUpperY <= chartArea.bottom) {
                            // Clamp to visible area
                            rUpperY = Math.max(rUpperY, chartArea.top);
                            rLowerY = Math.min(rLowerY, chartArea.bottom);
                            
                            // Fill zone
                            ctx.fillStyle = 'rgba(255, 68, 68, 0.08)';
                            ctx.fillRect(chartArea.left, rUpperY, chartArea.right - chartArea.left, rLowerY - rUpperY);
                            
                            // Upper line (24h high)
                            ctx.strokeStyle = 'rgba(255, 68, 68, 0.5)';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([2, 4]);
                            ctx.beginPath();
                            ctx.moveTo(chartArea.left, rUpperY);
                            ctx.lineTo(chartArea.right, rUpperY);
                            ctx.stroke();
                            
                            // Lower line (zone bottom)
                            ctx.beginPath();
                            ctx.moveTo(chartArea.left, rLowerY);
                            ctx.lineTo(chartArea.right, rLowerY);
                            ctx.stroke();
                            
                            // Label with symbol and price range + bounce count
                            var p2Symbol = window.pos2Data && window.pos2Data.symbol ? window.pos2Data.symbol : '';
                            var rTouchStr = resistanceTouches > 0 ? ' (' + resistanceTouches + ' bounces)' : ' (24h high)';
                            ctx.fillStyle = 'rgba(255, 68, 68, 0.8)';
                            ctx.font = '8px Orbitron';
                            ctx.textAlign = 'left';
                            ctx.fillText('🚫 RESISTANCE [' + resistanceLevel.toFixed(4) + ']' + rTouchStr + ' NO LONG', chartArea.left + 5, rUpperY + 12);
                        }
                        
                        // Draw SUPPORT SHADOW/CAUTION ZONE (cyan, very faint) - ABOVE the support zone
                        var sShadowUpperY = yScale.getPixelForValue(supportShadowUpper);
                        var sShadowLowerY = yScale.getPixelForValue(supportUpper);
                        if (sShadowUpperY >= chartArea.top && sShadowLowerY <= chartArea.bottom) {
                            sShadowUpperY = Math.max(sShadowUpperY, chartArea.top);
                            sShadowLowerY = Math.min(sShadowLowerY, chartArea.bottom);
                            
                            // Gradient from danger zone fading outward (upward)
                            var gradient = ctx.createLinearGradient(0, sShadowLowerY, 0, sShadowUpperY);
                            gradient.addColorStop(0, 'rgba(0, 200, 150, 0.06)');
                            gradient.addColorStop(1, 'rgba(0, 200, 150, 0.0)');
                            ctx.fillStyle = gradient;
                            ctx.fillRect(chartArea.left, sShadowUpperY, chartArea.right - chartArea.left, sShadowLowerY - sShadowUpperY);
                            
                            // Dashed line at shadow boundary
                            ctx.strokeStyle = 'rgba(0, 200, 150, 0.25)';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([1, 3]);
                            ctx.beginPath();
                            ctx.moveTo(chartArea.left, sShadowUpperY);
                            ctx.lineTo(chartArea.right, sShadowUpperY);
                            ctx.stroke();
                            ctx.setLineDash([]);
                            
                            // Label
                            ctx.fillStyle = 'rgba(0, 200, 150, 0.5)';
                            ctx.font = '7px Orbitron';
                            ctx.textAlign = 'left';
                            ctx.fillText('⚠️ CAUTION ZONE (reduce size)', chartArea.left + 5, sShadowUpperY + 10);
                        }
                        
                        // Draw SUPPORT ZONE (green, semi-transparent)
                        var sUpperY = yScale.getPixelForValue(supportUpper);
                        var sLowerY = yScale.getPixelForValue(supportLower);
                        if (sUpperY >= chartArea.top && sLowerY <= chartArea.bottom) {
                            // Clamp to visible area
                            sUpperY = Math.max(sUpperY, chartArea.top);
                            sLowerY = Math.min(sLowerY, chartArea.bottom);
                            
                            // Fill zone
                            ctx.fillStyle = 'rgba(0, 255, 136, 0.08)';
                            ctx.fillRect(chartArea.left, sUpperY, chartArea.right - chartArea.left, sLowerY - sUpperY);
                            
                            // Upper line (zone top)
                            ctx.strokeStyle = 'rgba(0, 255, 136, 0.5)';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([2, 4]);
                            ctx.beginPath();
                            ctx.moveTo(chartArea.left, sUpperY);
                            ctx.lineTo(chartArea.right, sUpperY);
                            ctx.stroke();
                            
                            // Lower line (24h low)
                            ctx.beginPath();
                            ctx.moveTo(chartArea.left, sLowerY);
                            ctx.lineTo(chartArea.right, sLowerY);
                            ctx.stroke();
                            
                            // Label with price range + bounce count
                            var sTouchStr = supportTouches > 0 ? ' (' + supportTouches + ' bounces)' : ' (24h low)';
                            ctx.fillStyle = 'rgba(0, 255, 136, 0.8)';
                            ctx.font = '8px Orbitron';
                            ctx.textAlign = 'left';
                            ctx.fillText('🚫 SUPPORT [' + supportLevel.toFixed(4) + ']' + sTouchStr + ' NO SHORT', chartArea.left + 5, sLowerY - 5);
                        }
                    }
                    
                    ctx.restore();
                }
            };
            
            // Update existing chart or create new one
            if (pos2Chart) {
                // Just update data and scales without destroying
                pos2Chart.data.datasets[0].data = highData;
                pos2Chart.data.datasets[1].data = lowData;
                pos2Chart.data.datasets[2].data = closeData;
                pos2Chart.options.scales.x.min = chartMinTime;
                pos2Chart.options.scales.x.max = chartMaxTime;
                pos2Chart.update('none'); // Update without animation
            } else {
                // Create new chart only on first load
                pos2Chart = new Chart(ctx, {
                    type: 'line',
                    plugins: [pos2LinesPlugin],
                    data: {
                        datasets: [
                            {
                                label: 'High',
                                data: highData,
                                borderColor: 'rgba(179, 102, 255, 0.15)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: false,
                                tension: 0.3
                            },
                            {
                                label: 'Low',
                                data: lowData,
                                borderColor: 'rgba(255, 102, 179, 0.15)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: '-1',
                                backgroundColor: 'rgba(179, 102, 255, 0.05)',
                                tension: 0.3
                            },
                            {
                                label: 'Price',
                                data: closeData,
                                borderColor: '#b366ff',
                                borderWidth: 2,
                                pointRadius: 0,
                                fill: false,
                                tension: 0.3
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: { enabled: false }
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: { unit: 'hour', displayFormats: { hour: 'HH:mm' } },
                                grid: { color: 'rgba(100, 100, 140, 0.1)' },
                                ticks: { color: '#666', font: { size: 9 } },
                                min: chartMinTime,
                                max: chartMaxTime,
                                offset: false,
                                bounds: 'data'
                            },
                            y: {
                                position: 'right',
                                grid: { color: 'rgba(100, 100, 140, 0.1)' },
                                ticks: { 
                                    color: '#888', 
                                    font: { size: 9 },
                                    callback: function(v) { return '$' + v.toFixed(4); }
                                }
                            }
                        }
                    }
                });
            }
        }
        
        function updatePriceChart(data) {
            if (!data || !data.candles || data.candles.length === 0) {
                console.log('[Position 1 Chart] No candle data received:', data);
                return;
            }
            
            var candles = data.candles;
            console.log('[Position 1 Chart] Updating with', candles.length, 'candles. Last close:', candles[candles.length - 1].c);
            var ctx = document.getElementById('priceChart').getContext('2d');
            
            // Update current price display
            var lastCandle = candles[candles.length - 1];
            var firstCandle = candles[0];
            var priceEl = document.getElementById('price-display');
            priceEl.textContent = '$' + lastCandle.c.toFixed(4);
            var priceChange = ((lastCandle.c - firstCandle.o) / firstCandle.o * 100).toFixed(2);
            priceEl.style.color = lastCandle.c >= firstCandle.o ? '#00ff88' : '#ff4444';
            
            // Update high/low/volume
            var highs = candles.map(c => c.h);
            var lows = candles.map(c => c.l);
            var volumes = candles.map(c => c.v);
            document.getElementById('chart-high').textContent = '$' + Math.max(...highs).toFixed(4);
            document.getElementById('chart-low').textContent = '$' + Math.min(...lows).toFixed(4);
            document.getElementById('chart-volume').textContent = volumes.reduce((a,b) => a+b, 0).toLocaleString();
            
            // Get user's timezone offset (server sends UTC timestamps)
            var tzOffset = new Date().getTimezoneOffset() * 60 * 1000; // in ms
            
            // Prepare candlestick data - timestamps are in UTC, display in local time
            var ohlcData = candles.map(c => ({
                x: new Date(c.t), // Chart.js handles timezone automatically
                o: c.o,
                h: c.h,
                l: c.l,
                c: c.c
            }));
            
            // Determine trend for beautiful gradient colors
            var isBullish = lastCandle.c >= firstCandle.o;
            var mainColor = isBullish ? '#00ff88' : '#ff4444';
            
            // Calculate timeframe interval in ms for future space
            var tfIntervalMs = 60000; // default 1m
            // Position 1 uses 15m candles - hardcode interval
            var tfIntervalMs = 15 * 60000; // 15m candles
            
            // Prepare smooth line data (close prices) for beautiful chart
            var closeData = candles.map(c => ({ x: new Date(c.t), y: c.c }));
            var highData = candles.map(c => ({ x: new Date(c.t), y: c.h }));
            var lowData = candles.map(c => ({ x: new Date(c.t), y: c.l }));
            
            // Calculate chart time range with extra space on the right for future prediction
            var lastCandleTime = new Date(lastCandle.t);
            var realNow = new Date(); // Current real time
            var firstTime = new Date(firstCandle.t);
            var futureSpace = 30; // Show 30 extra candle spaces for future prediction zone (same as Position 2)
            
            // Check if we're viewing historical data (last candle is more than 5 min old)
            var isHistorical = (realNow.getTime() - lastCandleTime.getTime()) > 5 * 60 * 1000;
            
            // For historical data, use last candle time; for live data, use current time
            var nowTime = isHistorical ? lastCandleTime : realNow;
            var chartMaxTime = new Date(lastCandleTime.getTime() + (tfIntervalMs * futureSpace));
            
            // Store in window scope for plugin access during updates
            window.chartNowTime = nowTime;
            window.chartLastCandleTime = lastCandleTime;
            window.chartMinTime = firstTime;
            window.chartMaxTime = chartMaxTime;
            window.pos1Candles = candles;  // Store candles for zone calculation
            var chartMinTime = firstTime;
            
            console.log('[P1 Chart] Range:', firstTime.toLocaleTimeString(), 'to', chartMaxTime.toLocaleTimeString());
            console.log('[P1 Chart] Last candle:', lastCandleTime.toLocaleTimeString(), 'Now:', nowTime.toLocaleTimeString());
            console.log('[P1 Chart] Future space ms:', tfIntervalMs * futureSpace, 'for', futureSpace, 'bars of', tfIntervalMs, 'ms each');
            
            // Create gradient function
            function createGradient(ctx, color) {
                var gradient = ctx.createLinearGradient(0, 0, 0, 180);
                if (color === '#00ff88') {
                    gradient.addColorStop(0, 'rgba(0, 255, 136, 0.5)');
                    gradient.addColorStop(0.3, 'rgba(0, 255, 136, 0.2)');
                    gradient.addColorStop(1, 'rgba(0, 255, 136, 0)');
                } else {
                    gradient.addColorStop(0, 'rgba(255, 68, 68, 0.5)');
                    gradient.addColorStop(0.3, 'rgba(255, 68, 68, 0.2)');
                    gradient.addColorStop(1, 'rgba(255, 68, 68, 0)');
                }
                return gradient;
            }
            
            if (priceChart) {
                // Update data without animation for snappy responsiveness
                priceChart.data.datasets[0].data = highData;
                priceChart.data.datasets[1].data = lowData;
                priceChart.data.datasets[2].data = closeData;
                priceChart.data.datasets[2].borderColor = mainColor;
                priceChart.data.datasets[2].pointBackgroundColor = mainColor;
                priceChart.data.datasets[2].backgroundColor = createGradient(ctx, mainColor);
                
                // Update time range for future space - LOCK THEM HARD
                priceChart.options.scales.x.min = chartMinTime;
                priceChart.options.scales.x.max = chartMaxTime;
                priceChart.options.scales.x.ticks.source = 'auto'; // Prevent auto scaling
                
                priceChart.update('none'); // No animation for instant response
                
                // Re-lock after update (Chart.js sometimes resets these)
                setTimeout(() => {
                    if (priceChart && priceChart.options.scales.x) {
                        priceChart.options.scales.x.min = chartMinTime;
                        priceChart.options.scales.x.max = chartMaxTime;
                    }
                }, 10);
                
                // Pulse effect on new data
                priceEl.style.transform = 'scale(1.15)';
                priceEl.style.textShadow = '0 0 30px ' + mainColor + ', 0 0 60px ' + mainColor;
                setTimeout(() => { 
                    priceEl.style.transform = 'scale(1)'; 
                    priceEl.style.textShadow = '';
                }, 300);
            } else {
                // Custom plugin to draw "NOW" line, future prediction zone, and ENTRY line
                var entryLineAnimation = 0; // For pulsing animation
                var futureZonePlugin = {
                    id: 'futureZone',
                    beforeDraw: function(chart) {
                        var ctx = chart.ctx;
                        var chartArea = chart.chartArea;
                        var xScale = chart.scales.x;
                        var yScale = chart.scales.y;
                        
                        // Get the x position of "now" using window scope (updated on every data refresh)
                        var currentNowTime = window.chartNowTime || new Date();
                        var nowX = xScale.getPixelForValue(currentNowTime);
                        
                        // === ENTRY PRICE LINE (when position is open) ===
                        var posData = window.lastDashboardData && window.lastDashboardData.open_position;
                        if (posData && posData.entry) {
                            var entryY = yScale.getPixelForValue(posData.entry);
                            var isLong = posData.side === 'LONG';
                            var entryColor = isLong ? '#00d4ff' : '#ff00ff';
                            
                            // Animate the glow intensity
                            entryLineAnimation = (entryLineAnimation + 0.05) % (Math.PI * 2);
                            var glowIntensity = 0.4 + Math.sin(entryLineAnimation) * 0.3;
                            
                            if (entryY >= chartArea.top && entryY <= chartArea.bottom) {
                                ctx.save();
                                
                                // Entry price label (small, on right side)
                                ctx.fillStyle = 'rgba(' + (isLong ? '0, 212, 255' : '255, 0, 255') + ', 0.9)';
                                ctx.font = '8px Orbitron';
                                ctx.textAlign = 'right';
                                ctx.fillText('ENTRY $' + posData.entry.toFixed(4), chartArea.right - 5, entryY - 3);
                                
                                // Draw entry lines based on whether we have entry_time
                                if (posData.entry_time) {
                                    var entryTime = new Date(posData.entry_time);
                                    // Recalculate entryX position each frame to ensure it moves with chart
                                    var entryX = xScale.getPixelForValue(entryTime);
                                    
                                    // Only draw if entry is within visible chart area
                                    if (entryX >= chartArea.left && entryX <= chartArea.right) {
                                        // Vertical line at entry point (full height)
                                        ctx.strokeStyle = 'rgba(' + (isLong ? '0, 212, 255' : '255, 0, 255') + ', ' + glowIntensity + ')';
                                        ctx.lineWidth = 1;
                                        ctx.setLineDash([3, 3]);
                                        ctx.beginPath();
                                        ctx.moveTo(entryX, chartArea.top);
                                        ctx.lineTo(entryX, chartArea.bottom);
                                        ctx.stroke();
                                        
                                        // Redraw horizontal entry line starting FROM entry point (not from left edge)
                                        ctx.strokeStyle = 'rgba(' + (isLong ? '0, 212, 255' : '255, 0, 255') + ', ' + glowIntensity + ')';
                                        ctx.lineWidth = 1;
                                        ctx.setLineDash([]);
                                        ctx.beginPath();
                                        ctx.moveTo(entryX, entryY);  // Start from entry point
                                        ctx.lineTo(chartArea.right, entryY);  // To the right edge
                                        ctx.stroke();
                                        
                                        // Entry marker dot at exact intersection
                                        ctx.beginPath();
                                        ctx.arc(entryX, entryY, 4 + Math.sin(entryLineAnimation) * 2, 0, Math.PI * 2);
                                        ctx.fillStyle = entryColor;
                                        ctx.fill();
                                        ctx.strokeStyle = '#fff';
                                        ctx.lineWidth = 1;
                                        ctx.stroke();
                                        
                                        // Entry time label on the vertical line
                                        ctx.fillStyle = 'rgba(' + (isLong ? '0, 212, 255' : '255, 0, 255') + ', 0.8)';
                                        ctx.font = '7px Orbitron';
                                        ctx.textAlign = 'center';
                                        ctx.fillText(entryTime.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}), entryX, chartArea.top + 10);
                                    }
                                } else {
                                    // No entry_time - draw full-width horizontal line
                                    ctx.strokeStyle = 'rgba(' + (isLong ? '0, 212, 255' : '255, 0, 255') + ', ' + glowIntensity + ')';
                                    ctx.lineWidth = 1;
                                    ctx.setLineDash([]);
                                    ctx.beginPath();
                                    ctx.moveTo(chartArea.left, entryY);
                                    ctx.lineTo(chartArea.right, entryY);
                                    ctx.stroke();
                                }
                                
                                // SL line (thin, dashed)
                                if (posData.stop_loss) {
                                    var slY = yScale.getPixelForValue(posData.stop_loss);
                                    if (slY >= chartArea.top && slY <= chartArea.bottom) {
                                        ctx.strokeStyle = 'rgba(255, 68, 68, 0.5)';
                                        ctx.lineWidth = 1;
                                        ctx.setLineDash([3, 3]);
                                        ctx.beginPath();
                                        ctx.moveTo(chartArea.left, slY);
                                        ctx.lineTo(chartArea.right, slY);
                                        ctx.stroke();
                                        
                                        ctx.fillStyle = 'rgba(255, 68, 68, 0.7)';
                                        ctx.font = '7px Orbitron';
                                        ctx.textAlign = 'right';
                                        ctx.fillText('SL', chartArea.right - 5, slY - 2);
                                    }
                                }
                                
                                // TP1 line (thin, dashed)
                                if (posData.tp1) {
                                    var tp1Y = yScale.getPixelForValue(posData.tp1);
                                    if (tp1Y >= chartArea.top && tp1Y <= chartArea.bottom) {
                                        ctx.strokeStyle = 'rgba(0, 255, 136, 0.4)';
                                        ctx.lineWidth = 1;
                                        ctx.setLineDash([3, 3]);
                                        ctx.beginPath();
                                        ctx.moveTo(chartArea.left, tp1Y);
                                        ctx.lineTo(chartArea.right, tp1Y);
                                        ctx.stroke();
                                        
                                        ctx.fillStyle = 'rgba(0, 255, 136, 0.7)';
                                        ctx.font = '7px Orbitron';
                                        ctx.textAlign = 'right';
                                        ctx.fillText('TP1', chartArea.right - 5, tp1Y - 2);
                                    }
                                }
                                
                                // TP2 line (thin, dashed)
                                if (posData.tp2) {
                                    var tp2Y = yScale.getPixelForValue(posData.tp2);
                                    if (tp2Y >= chartArea.top && tp2Y <= chartArea.bottom) {
                                        ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
                                        ctx.lineWidth = 1;
                                        ctx.setLineDash([3, 3]);
                                        ctx.beginPath();
                                        ctx.moveTo(chartArea.left, tp2Y);
                                        ctx.lineTo(chartArea.right, tp2Y);
                                        ctx.stroke();
                                        
                                        ctx.fillStyle = 'rgba(0, 255, 136, 0.5)';
                                        ctx.font = '7px Orbitron';
                                        ctx.textAlign = 'right';
                                        ctx.fillText('TP2', chartArea.right - 5, tp2Y - 2);
                                    }
                                }
                                
                                ctx.restore();
                            }
                        }
                        
                        // Position 2 lines are now drawn on a separate dedicated chart
                        
                        // Draw shaded future zone
                        if (nowX < chartArea.right) {
                            ctx.save();
                            ctx.fillStyle = 'rgba(0, 212, 255, 0.03)';
                            ctx.fillRect(nowX, chartArea.top, chartArea.right - nowX, chartArea.bottom - chartArea.top);
                            
                            // Draw diagonal stripes pattern
                            ctx.strokeStyle = 'rgba(0, 212, 255, 0.05)';
                            ctx.lineWidth = 1;
                            var stripeGap = 15;
                            for (var i = 0; i < (chartArea.right - nowX) + (chartArea.bottom - chartArea.top); i += stripeGap) {
                                ctx.beginPath();
                                ctx.moveTo(nowX + i, chartArea.top);
                                ctx.lineTo(nowX + i - (chartArea.bottom - chartArea.top), chartArea.bottom);
                                ctx.stroke();
                            }
                            
                            // Draw "PREDICTION ZONE" label
                            ctx.fillStyle = 'rgba(0, 212, 255, 0.3)';
                            ctx.font = '9px Orbitron';
                            ctx.textAlign = 'center';
                            var labelX = nowX + (chartArea.right - nowX) / 2;
                            ctx.fillText('PREDICTION ZONE', labelX, chartArea.top + 15);
                            ctx.restore();
                        }
                        
                        // === RESISTANCE & SUPPORT ZONES (Position 1) ===
                        var candleData = window.pos1Candles;
                        if (candleData && candleData.length > 0) {
                            // Calculate 24h high and low
                            var highs = candleData.map(c => c.h);
                            var lows = candleData.map(c => c.l);
                            var high24h = Math.max(...highs);
                            var low24h = Math.min(...lows);
                            
                            // Calculate ATR for zone width
                            var trValues = [];
                            for (var i = 1; i < Math.min(14, candleData.length); i++) {
                                var tr = Math.max(
                                    candleData[i].h - candleData[i].l,
                                    Math.abs(candleData[i].h - candleData[i-1].c),
                                    Math.abs(candleData[i].l - candleData[i-1].c)
                                );
                                trValues.push(tr);
                            }
                            var atr = trValues.length > 0 ? trValues.reduce((a,b) => a+b, 0) / trValues.length : (high24h - low24h) * 0.1;
                            
                            // Zone width = ATR * 1.5 (clamped between 1% and 5%)
                            var currentPrice = candleData[candleData.length - 1].c;
                            var zoneWidth = atr * 0.5;  // REDUCED from 1.5x to 0.5x ATR
                            var minZone = currentPrice * 0.01;
                            var maxZone = currentPrice * 0.05;
                            zoneWidth = Math.max(minZone, Math.min(maxZone, zoneWidth));
                            
                            // 4 Price Levels
                            var resistanceUpper = high24h;
                            var resistanceLower = high24h - zoneWidth;
                            var supportUpper = low24h + zoneWidth;
                            var supportLower = low24h;
                            
                            ctx.save();
                            
                            // Draw RESISTANCE ZONE (red, semi-transparent)
                            var rUpperY = yScale.getPixelForValue(resistanceUpper);
                            var rLowerY = yScale.getPixelForValue(resistanceLower);
                            if (rLowerY >= chartArea.top && rUpperY <= chartArea.bottom) {
                                // Clamp to visible area
                                rUpperY = Math.max(rUpperY, chartArea.top);
                                rLowerY = Math.min(rLowerY, chartArea.bottom);
                                
                                // Fill zone
                                ctx.fillStyle = 'rgba(255, 68, 68, 0.08)';
                                ctx.fillRect(chartArea.left, rUpperY, chartArea.right - chartArea.left, rLowerY - rUpperY);
                                
                                // Upper line (24h high)
                                ctx.strokeStyle = 'rgba(255, 68, 68, 0.5)';
                                ctx.lineWidth = 1;
                                ctx.setLineDash([2, 4]);
                                ctx.beginPath();
                                ctx.moveTo(chartArea.left, rUpperY);
                                ctx.lineTo(chartArea.right, rUpperY);
                                ctx.stroke();
                                
                                // Lower line (zone bottom)
                                ctx.beginPath();
                                ctx.moveTo(chartArea.left, rLowerY);
                                ctx.lineTo(chartArea.right, rLowerY);
                                ctx.stroke();
                                
                                // Label with price range
                                ctx.fillStyle = 'rgba(255, 68, 68, 0.8)';
                                ctx.font = '8px Orbitron';
                                ctx.textAlign = 'left';
                                ctx.fillText('🚫 RESISTANCE [' + resistanceLower.toFixed(4) + '-' + resistanceUpper.toFixed(4) + '] NO LONG', chartArea.left + 5, rUpperY + 12);
                            }
                            
                            // Draw SUPPORT ZONE (green, semi-transparent)
                            var sUpperY = yScale.getPixelForValue(supportUpper);
                            var sLowerY = yScale.getPixelForValue(supportLower);
                            if (sUpperY >= chartArea.top && sLowerY <= chartArea.bottom) {
                                // Clamp to visible area
                                sUpperY = Math.max(sUpperY, chartArea.top);
                                sLowerY = Math.min(sLowerY, chartArea.bottom);
                                
                                // Fill zone
                                ctx.fillStyle = 'rgba(0, 255, 136, 0.08)';
                                ctx.fillRect(chartArea.left, sUpperY, chartArea.right - chartArea.left, sLowerY - sUpperY);
                                
                                // Upper line (zone top)
                                ctx.strokeStyle = 'rgba(0, 255, 136, 0.5)';
                                ctx.lineWidth = 1;
                                ctx.setLineDash([2, 4]);
                                ctx.beginPath();
                                ctx.moveTo(chartArea.left, sUpperY);
                                ctx.lineTo(chartArea.right, sUpperY);
                                ctx.stroke();
                                
                                // Lower line (24h low)
                                ctx.beginPath();
                                ctx.moveTo(chartArea.left, sLowerY);
                                ctx.lineTo(chartArea.right, sLowerY);
                                ctx.stroke();
                                
                                // Label with price range
                                ctx.fillStyle = 'rgba(0, 255, 136, 0.8)';
                                ctx.font = '8px Orbitron';
                                ctx.textAlign = 'left';
                                ctx.fillText('🚫 SUPPORT [' + supportLower.toFixed(4) + '-' + supportUpper.toFixed(4) + '] NO SHORT', chartArea.left + 5, sLowerY - 5);
                            }
                            
                            ctx.restore();
                        }
                        
                        // Draw vertical "NOW" line
                        if (nowX >= chartArea.left && nowX <= chartArea.right) {
                            ctx.save();
                            ctx.strokeStyle = '#00ff88';
                            ctx.lineWidth = 2;
                            ctx.setLineDash([5, 3]);
                            ctx.beginPath();
                            ctx.moveTo(nowX, chartArea.top);
                            ctx.lineTo(nowX, chartArea.bottom);
                            ctx.stroke();
                            
                            // "NOW" label
                            ctx.fillStyle = '#00ff88';
                            ctx.font = 'bold 9px Orbitron';
                            ctx.textAlign = 'center';
                            ctx.fillText('NOW', nowX, chartArea.bottom + 12);
                            ctx.restore();
                        }
                    }
                };
                
                priceChart = new Chart(ctx, {
                    type: 'line',
                    plugins: [futureZonePlugin],
                    data: {
                        datasets: [
                            // High range (subtle glow)
                            {
                                label: 'High',
                                data: highData,
                                borderColor: 'rgba(0, 255, 136, 0.15)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: false,
                                tension: 0.4
                            },
                            // Low range (fill area between high and low)
                            {
                                label: 'Low',
                                data: lowData,
                                borderColor: 'rgba(255, 68, 68, 0.15)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: '-1',
                                backgroundColor: 'rgba(0, 212, 255, 0.03)',
                                tension: 0.4
                            },
                            // Main price line with glow effect
                            {
                                label: 'Price',
                                data: closeData,
                                borderColor: mainColor,
                                borderWidth: 3,
                                pointRadius: 0,
                                pointHoverRadius: 8,
                                pointHoverBackgroundColor: mainColor,
                                pointHoverBorderColor: '#fff',
                                pointHoverBorderWidth: 3,
                                fill: true,
                                backgroundColor: createGradient(ctx, mainColor),
                                tension: 0.4,
                                cubicInterpolationMode: 'monotone'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: false,
                        interaction: {
                            mode: 'index',
                            intersect: false
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: { enabled: false },
                            zoom: {
                                zoom: { enabled: false },
                                pan: { enabled: false }
                            }
                        },
                        layout: {
                            padding: { right: 0, left: 0 }
                        },
                        onHover: function(event, activeElements) {
                            // Lock time bounds on every interaction
                            if (priceChart && priceChart.options.scales.x) {
                                priceChart.options.scales.x.min = chartMinTime;
                                priceChart.options.scales.x.max = chartMaxTime;
                            }
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: { 
                                    unit: 'minute', 
                                    displayFormats: { minute: 'HH:mm', hour: 'HH:mm' },
                                    tooltipFormat: 'HH:mm:ss'
                                },
                                min: chartMinTime,
                                max: chartMaxTime,
                                offset: false,
                                grid: { 
                                    color: 'rgba(0, 212, 255, 0.08)',
                                    drawBorder: false
                                },
                                ticks: { 
                                    color: '#555', 
                                    maxTicksLimit: 10,
                                    font: { family: 'Orbitron', size: 9 },
                                    source: 'auto'
                                }
                            },
                            y: {
                                position: 'right',
                                grid: { 
                                    color: 'rgba(0, 212, 255, 0.08)',
                                    drawBorder: false
                                },
                                ticks: { 
                                    color: '#555', 
                                    callback: v => '$' + v.toFixed(4),
                                    font: { family: 'Orbitron', size: 9 }
                                }
                            }
                        }
                    }
                });
                
                console.log('[P1 Chart Created] min:', priceChart.options.scales.x.min.toLocaleTimeString(), 'max:', priceChart.options.scales.x.max.toLocaleTimeString());
                console.log('[P1 Chart Created] Dataset 2 (close) points:', priceChart.data.datasets[2].data.length);
                
                // Add mouse/touch move listener for crosshair and tooltip
                var canvas = document.getElementById('priceChart');
                var chartWrapper = document.getElementById('chart-wrapper');
                var floatTooltip = document.getElementById('chart-float-tooltip');
                var crosshairH = document.getElementById('crosshair-h');
                var crosshairV = document.getElementById('crosshair-v');
                var priceLabel = document.getElementById('price-label');
                var timeLabel = document.getElementById('time-label');
                
                function handleChartInteraction(e) {
                    var rect = canvas.getBoundingClientRect();
                    let clientX, clientY;
                    
                    // Handle both mouse and touch events
                    if (e.touches && e.touches.length > 0) {
                        clientX = e.touches[0].clientX;
                        clientY = e.touches[0].clientY;
                    } else {
                        clientX = e.clientX;
                        clientY = e.clientY;
                    }
                    
                    var x = clientX - rect.left;
                    var y = clientY - rect.top;
                    
                    // Show crosshair
                    crosshairH.style.display = 'block';
                    crosshairH.style.top = y + 'px';
                    crosshairV.style.display = 'block';
                    crosshairV.style.left = x + 'px';
                    
                    var elements = priceChart.getElementsAtEventForMode(e, 'index', { intersect: false }, false);
                    
                    if (elements.length > 0) {
                        var dataIndex = elements[0].index;
                        var dataset = priceChart.data.datasets[0].data;
                        // Use currentVisibleCandles for accurate hover data
                        var candle = currentVisibleCandles[dataIndex];
                        if (dataset && dataset[dataIndex] && candle) {
                            var d = dataset[dataIndex];
                            var date = new Date(candle.t);
                            var change = ((candle.c - candle.o) / candle.o * 100);
                            
                            // Date + time format: "Jan 12, 14:35"
                            var dateStr = date.toLocaleDateString('en-US', {month: 'short', day: 'numeric'}) + ', ' + date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
                            
                            // Show price label on crosshair
                            priceLabel.style.display = 'block';
                            priceLabel.style.top = y + 'px';
                            priceLabel.textContent = '$' + candle.c.toFixed(4);
                            
                            // Show time label on crosshair - include date
                            timeLabel.style.display = 'block';
                            timeLabel.style.left = x + 'px';
                            timeLabel.textContent = dateStr;
                            
                            // Update floating tooltip with full date
                            document.getElementById('tt-time').textContent = date.toLocaleString('en-US', {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'});
                            document.getElementById('tt-open').textContent = '$' + candle.o.toFixed(4);
                            document.getElementById('tt-high').textContent = '$' + candle.h.toFixed(4);
                            document.getElementById('tt-low').textContent = '$' + candle.l.toFixed(4);
                            document.getElementById('tt-close').textContent = '$' + candle.c.toFixed(4);
                            document.getElementById('tt-close').className = 'tt-value ' + (candle.c >= candle.o ? 'up' : 'down');
                            var ttChange = document.getElementById('tt-change');
                            ttChange.textContent = (change >= 0 ? '▲ +' : '▼ ') + change.toFixed(2) + '%';
                            ttChange.style.color = change >= 0 ? '#00ff88' : '#ff4444';
                            
                            // Position tooltip near cursor with smooth animation
                            let tooltipX = x + 20;
                            let tooltipY = y - 100;
                            
                            // Keep tooltip within canvas bounds
                            if (tooltipX + 220 > rect.width) {
                                tooltipX = x - 240;
                            }
                            if (tooltipY < 10) {
                                tooltipY = y + 20;
                            }
                            if (tooltipY + 180 > rect.height) {
                                tooltipY = rect.height - 190;
                            }
                            
                            floatTooltip.style.left = tooltipX + 'px';
                            floatTooltip.style.top = tooltipY + 'px';
                            floatTooltip.style.display = 'block';
                            floatTooltip.style.opacity = '1';
                        }
                    }
                }
                
                function hideChartElements() {
                    floatTooltip.style.display = 'none';
                    crosshairH.style.display = 'none';
                    crosshairV.style.display = 'none';
                    priceLabel.style.display = 'none';
                    timeLabel.style.display = 'none';
                }
                
                // Mouse events
                canvas.addEventListener('mousemove', handleChartInteraction);
                canvas.addEventListener('mouseleave', hideChartElements);
                
                // Touch events for mobile
                canvas.addEventListener('touchmove', function(e) {
                    e.preventDefault();
                    handleChartInteraction(e);
                }, { passive: false });
                canvas.addEventListener('touchstart', function(e) {
                    handleChartInteraction(e);
                });
                canvas.addEventListener('touchend', hideChartElements);
            }
        }
        
        function updateDashboard(data) {
            console.log('updateDashboard called with data:', data ? 'yes' : 'no');
            
            // Store for position checks (used by switchToSymbol)
            window.lastDashboardData = data;
            
            try {
                // Helper function to safely set text
                function setText(id, value) {
                    var el = document.getElementById(id);
                    if (el) el.textContent = value;
                }
                
                function setClass(id, className) {
                    var el = document.getElementById(id);
                    if (el) el.className = className;
                }
                
                // Status
                if (data.status) {
                    var statusDot = document.getElementById('status-dot');
                    var statusText = document.getElementById('status-text');
                    if (data.status.connected) {
                        if (data.status.paused) {
                            if (statusDot) statusDot.className = 'status-dot paused';
                            if (statusText) statusText.textContent = 'PAUSED';
                        } else {
                            if (statusDot) statusDot.className = 'status-dot online';
                            if (statusText) statusText.textContent = 'RUNNING';
                        }
                    } else {
                        if (statusDot) statusDot.className = 'status-dot offline';
                        if (statusText) statusText.textContent = 'OFFLINE';
                    }
                    // Update chart symbol header - use STABLE chart symbol (matches what chart is actually showing)
                    var chartSymbol = document.getElementById('chart-symbol');
                    var displaySymbol = getStableChartSymbol();
                    
                    if (chartSymbol && displaySymbol) {
                        // Format symbol nicely (LINKUSDT -> LINK/USDT)
                        var sym = displaySymbol;
                        // Clean up various formats
                        sym = sym.replace('/USDT:USDT', '/USDT').replace(':USDT', '');
                        if (sym.indexOf('/') === -1 && sym.endsWith('USDT')) {
                            sym = sym.replace('USDT', '/USDT');
                        }
                        // Add source indicator: 📊 for position, 🔍 for scanner
                        var sourceIcon = window._chartSymbolSource === 'position' ? '📊 ' : '🔍 ';
                        chartSymbol.textContent = sourceIcon + sym;
                        // Keep global currentSymbol in sync with what's displayed
                        currentSymbol = displaySymbol;
                    }
                }
                
                // Balance
                var bal = data.balance || {};
                setText('balance', '$' + (bal.current || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}));
                var changePct = bal.change_pct || 0;
                var changeEl = document.getElementById('balance-change');
                if (changeEl) {
                    changeEl.textContent = (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%';
                    changeEl.style.color = changePct >= 0 ? '#00ff88' : '#ff4444';
                }
                
                // UTA Margin Info (for Bybit Unified Trading Account)
                var status = data.status || {};
                var utaCard = document.getElementById('uta-margin-card');
                if (utaCard) {
                    // Always show available margin
                    var availMargin = status.uta_available_margin || status.balance || 0;
                    setText('uta-available', '$' + availMargin.toFixed(2));
                    
                    // Color based on amount
                    var availEl = document.getElementById('uta-available');
                    if (availEl) {
                        availEl.style.color = availMargin < 10 ? '#ff4444' : 
                                              availMargin < 50 ? '#ff9800' : '#00ff88';
                    }
                    
                    // Show borrowed/LTV only if there's a loan
                    var borrowedEl = document.getElementById('uta-borrowed');
                    var ltvEl = document.getElementById('uta-ltv');
                    if (status.uta_borrowed > 0) {
                        if (borrowedEl) {
                            borrowedEl.style.display = 'block';
                            borrowedEl.textContent = 'Borrowed: $' + (status.uta_borrowed || 0).toFixed(0);
                        }
                        if (ltvEl) {
                            ltvEl.style.display = 'block';
                            var ltv = status.uta_ltv || 0;
                            ltvEl.textContent = 'LTV: ' + ltv.toFixed(1) + '%';
                            ltvEl.style.color = ltv > 80 ? '#ff4444' : ltv > 60 ? '#ff9800' : '#00ff88';
                        }
                    } else {
                        if (borrowedEl) borrowedEl.style.display = 'none';
                        if (ltvEl) ltvEl.style.display = 'none';
                    }
                }
                
                // P&L
                var pnl = data.pnl || {};
                var todayPnl = pnl.today || 0;
                var todayEl = document.getElementById('today-pnl');
                if (todayEl) {
                    todayEl.textContent = (todayPnl >= 0 ? '+$' : '-$') + Math.abs(todayPnl).toFixed(2);
                    todayEl.className = 'value ' + (todayPnl >= 0 ? 'positive' : 'negative');
                }
                
                var totalPnl = pnl.total || 0;
                var totalEl = document.getElementById('total-pnl');
                if (totalEl) {
                    totalEl.textContent = (totalPnl >= 0 ? '+$' : '-$') + Math.abs(totalPnl).toFixed(2);
                    totalEl.className = 'value ' + (totalPnl >= 0 ? 'positive' : 'negative');
                }
                
                // Win Rate
                setText('win-rate', ((pnl.win_rate || 0) * 100).toFixed(1) + '%');
                setText('win-loss', (pnl.winning || 0) + 'W / ' + ((pnl.trades || 0) - (pnl.winning || 0)) + 'L');
                setText('total-trades', pnl.trades || 0);
                
                // ========== ENHANCED CURRENT SIGNAL ==========
                var signal = data.current_signal || {};
                var pos = data.open_position;
                var sigBox = document.getElementById('signal-box-main');
                var sigValue = document.getElementById('signal-direction-text');
                var confidenceBar = document.getElementById('confidence-bar');
                var confidencePct = document.getElementById('confidence-pct');
                
                // Update signal strength bars
                function updateSignalStrength(confidence) {
                    var bars = 5;
                    var activeCount = Math.round((confidence || 0) * bars);
                    for (var i = 1; i <= bars; i++) {
                        var bar = document.getElementById('str-' + i);
                        if (bar) {
                            if (i <= activeCount) {
                                bar.classList.add('active');
                                if (confidence < 0.4) bar.classList.add('low');
                                else if (confidence < 0.7) bar.classList.add('medium');
                                else bar.classList.add('high');
                            } else {
                                bar.className = 'strength-bar';
                            }
                        }
                    }
                }
                
                if (sigBox && sigValue) {
                    if (signal.direction) {
                        // Active signal
                        var dirClass = signal.direction === 'LONG' ? 'long' : signal.direction === 'SHORT' ? 'short' : 'neutral';
                        sigBox.className = 'signal-box ' + dirClass + ' active-signal';
                        sigValue.textContent = signal.direction;
                        
                        // Confidence gauge
                        var conf = signal.confidence || 0;
                        if (confidenceBar) {
                            confidenceBar.style.width = (conf * 100) + '%';
                            confidenceBar.style.background = conf < 0.5 ? 'linear-gradient(90deg, #ff4444, #ffaa00)' : 
                                                             conf < 0.7 ? 'linear-gradient(90deg, #ffaa00, #00d4ff)' : 
                                                             'linear-gradient(90deg, #00d4ff, #00ff88)';
                        }
                        if (confidencePct) confidencePct.textContent = (conf * 100).toFixed(0) + '%';
                        updateSignalStrength(conf);
                        
                        setText('sig-entry', signal.entry ? '$' + signal.entry.toFixed(4) : '--');
                        setText('sig-sl', signal.stop_loss ? '$' + signal.stop_loss.toFixed(4) : '--');
                        setText('sig-tp', signal.tp1 ? '$' + signal.tp1.toFixed(4) : '--');
                    } else if (pos && pos.symbol) {
                        // In position - show active status
                        var posClass = pos.side === 'LONG' ? 'long' : 'short';
                        sigBox.className = 'signal-box ' + posClass + ' active-signal';
                        sigValue.textContent = pos.side + ' ACTIVE';
                        
                        if (confidenceBar) { confidenceBar.style.width = '100%'; confidenceBar.style.background = posClass === 'long' ? '#00ff88' : '#ff4444'; }
                        if (confidencePct) confidencePct.textContent = 'IN TRADE';
                        updateSignalStrength(1);
                        
                        setText('sig-entry', '$' + pos.entry.toFixed(4));
                        setText('sig-sl', pos.stop_loss ? '$' + pos.stop_loss.toFixed(4) : '--');
                        setText('sig-tp', pos.tp1 ? '$' + pos.tp1.toFixed(4) : '--');
                    } else {
                        // No signal, no position
                        sigBox.className = 'signal-box neutral';
                        sigValue.textContent = 'WAITING';
                        
                        if (confidenceBar) { confidenceBar.style.width = '0%'; }
                        if (confidencePct) confidencePct.textContent = '--';
                        updateSignalStrength(0);
                        
                        setText('sig-entry', '--');
                        setText('sig-sl', '--');
                        setText('sig-tp', '--');
                    }
                }
                
                // ========== DUAL POSITION SLOTS - Always Show Both ==========
                // Get position slot elements
                var pos1Content = document.getElementById('pos1-content');
                var pos2Content = document.getElementById('pos2-content');
                // Buttons are now rendered inside dynamic content - no need for static button references
                
                // Get all open positions from data
                var positions = [];
                if (data.open_position && data.open_position.symbol) {
                    positions.push(data.open_position);
                }
                if (data.additional_positions && Array.isArray(data.additional_positions)) {
                    positions = positions.concat(data.additional_positions);
                }
                
                console.log('[Dashboard] Total positions collected:', positions.length);
                
                // Store symbols globally for close buttons
                window.pos1Symbol = null;
                window.pos2Symbol = null;
                
                // Helper function to render position content
                function renderPositionSlot(pos, slotNum) {
                    var contentEl = slotNum === 1 ? pos1Content : pos2Content;
                    
                    if (!pos || !pos.symbol) {
                        // Empty slot - show waiting state and hide buttons
                        if (slotNum === 1) {
                            window.pos1Symbol = null;
                            var btns1 = document.getElementById('pos1-buttons');
                            if (btns1) btns1.style.display = 'none';
                        } else {
                            window.pos2Symbol = null;
                            var btns2 = document.getElementById('pos2-buttons');
                            if (btns2) btns2.style.display = 'none';
                        }
                        
                        if (contentEl) {
                            contentEl.innerHTML = `
                                <div class="empty-slot" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 160px; color: #555; text-align: center;">
                                    <div style="font-size: 36px; margin-bottom: 10px; opacity: 0.4;">📊</div>
                                    <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 2px;">Waiting for Trade</div>
                                    <div style="font-size: 11px; color: #444; margin-top: 8px;">AI Scanning...</div>
                                </div>
                            `;
                        }
                        return;
                    }
                    
                    // Store symbol and show buttons
                    if (slotNum === 1) {
                        window.pos1Symbol = pos.symbol;
                        var btns1 = document.getElementById('pos1-buttons');
                        if (btns1) btns1.style.display = 'flex';
                    } else {
                        window.pos2Symbol = pos.symbol;
                        var btns2 = document.getElementById('pos2-buttons');
                        if (btns2) btns2.style.display = 'flex';
                    }
                    
                    // Calculate position of current price on the scale
                    var entry = pos.entry || 0;
                    var current = pos.current_price || pos.entry || 0;
                    var sl = pos.stop_loss || entry;
                    var tp1 = pos.tp1 || entry;
                    var tp2 = pos.tp2 || entry;
                    var tp3 = pos.tp3 || entry;
                    
                    // Determine scale range
                    var isLong = pos.side === 'LONG';
                    var minPrice, maxPrice;
                    if (isLong) {
                        minPrice = Math.min(sl, entry);
                        maxPrice = Math.max(tp3, tp2, tp1, current);
                    } else {
                        minPrice = Math.min(tp3, tp2, tp1, current);
                        maxPrice = Math.max(sl, entry);
                    }
                    
                    var range = maxPrice - minPrice || 1;
                    var slPercent = ((sl - minPrice) / range) * 100;
                    var entryPercent = ((entry - minPrice) / range) * 100;
                    var tp1Percent = ((tp1 - minPrice) / range) * 100;
                    var tp2Percent = ((tp2 - minPrice) / range) * 100;
                    var tp3Percent = ((tp3 - minPrice) / range) * 100;
                    var currentPercent = ((current - minPrice) / range) * 100;
                    
                    // Calculate water flow width (from entry toward current price)
                    var flowStart = Math.min(entryPercent, currentPercent);
                    var flowEnd = Math.max(entryPercent, currentPercent);
                    var flowWidth = flowEnd - flowStart;
                    var isProfit = (current - entry) * (isLong ? 1 : -1) >= 0;
                    
                    var slotHTML = `
                        <!-- Position Header - Vibrant -->
                        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 14px; flex-wrap: wrap;">
                            <span class="position-side-badge ${pos.side === 'LONG' ? 'long' : 'short'}" style="font-size: 1.2rem; padding: 8px 18px; background: ${pos.side === 'LONG' ? 'linear-gradient(135deg, rgba(0,255,136,0.4), rgba(0,200,100,0.3))' : 'linear-gradient(135deg, rgba(255,68,68,0.4), rgba(200,50,50,0.3))'}; border: 1px solid ${pos.side === 'LONG' ? 'rgba(0,255,136,0.6)' : 'rgba(255,100,100,0.6)'}; border-radius: 10px; text-shadow: 0 0 10px ${pos.side === 'LONG' ? 'rgba(0,255,136,0.8)' : 'rgba(255,68,68,0.8)'}; box-shadow: 0 0 20px ${pos.side === 'LONG' ? 'rgba(0,255,136,0.3)' : 'rgba(255,68,68,0.3)'};">${pos.side}</span>
                            <span style="color: #fff; font-size: 1.4rem; font-weight: bold; text-shadow: 0 0 15px rgba(255,255,255,0.5), 0 2px 4px rgba(0,0,0,0.5);">${pos.symbol}</span>
                            <span class="pnl-value-large ${(pos.pnl || 0) >= 0 ? 'profit' : 'loss'}" style="font-size: 1.5rem; margin-left: auto; font-weight: bold; text-shadow: 0 0 20px ${(pos.pnl || 0) >= 0 ? 'rgba(0,255,136,0.8)' : 'rgba(255,68,68,0.8)'}; animation: ${(pos.pnl || 0) >= 0 ? 'profitGlow' : 'lossGlow'} 2s ease-in-out infinite;">
                                ${(pos.pnl || 0) >= 0 ? '+$' : '-$'}${Math.abs(pos.pnl || 0).toFixed(2)}
                            </span>
                        </div>
                        
                        <!-- Water Flow Scale - Vibrant -->
                        <div style="position: relative; height: 70px; background: linear-gradient(135deg, rgba(5,10,20,0.8) 0%, rgba(15,20,35,0.7) 50%, rgba(10,15,25,0.8) 100%); border-radius: 12px; border: 1px solid rgba(100,200,255,0.3); overflow: hidden; margin: 12px 0; box-shadow: inset 0 2px 15px rgba(0,0,0,0.6), 0 0 25px rgba(0,150,255,0.15);">
                            
                            <!-- Background wave pattern -->
                            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: repeating-linear-gradient(90deg, transparent, transparent 25px, rgba(100,180,255,0.04) 25px, rgba(100,180,255,0.04) 50px); pointer-events: none;"></div>
                            
                            <!-- Vibrant gradient base -->
                            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(90deg, rgba(255,100,100,0.1), rgba(100,200,255,0.1), rgba(100,255,200,0.1), rgba(255,200,100,0.1)); opacity: 0.6;"></div>
                            
                            <!-- Water Flow Animation -->
                            <div class="water-flow-bar ${isProfit ? 'profit' : 'loss'}" style="left: ${flowStart}%; width: ${flowWidth}%; top: 0; border-radius: 6px;"></div>
                            
                            <!-- SL Marker - Vibrant -->
                            <div style="position: absolute; left: ${slPercent}%; top: 0; height: 100%; width: 4px; background: linear-gradient(to bottom, #ff5555, #ff3333, rgba(255,68,68,0.3)); z-index: 5; box-shadow: 0 0 15px rgba(255,68,68,0.8), 0 0 30px rgba(255,68,68,0.4);"></div>
                            <div style="position: absolute; left: ${slPercent}%; top: 6px; font-size: 13px; color: #ff7777; transform: translateX(-50%); font-weight: bold; text-shadow: 0 0 10px rgba(255,68,68,0.9), 0 0 20px rgba(255,0,0,0.5);">SL</div>
                            
                            <!-- Entry Marker - Vibrant -->
                            <div style="position: absolute; left: ${entryPercent}%; top: 0; height: 100%; width: 5px; background: linear-gradient(to bottom, #ffff44, #ffcc00, rgba(255,255,0,0.4)); z-index: 6; box-shadow: 0 0 18px rgba(255,255,0,0.9), 0 0 35px rgba(255,200,0,0.5);"></div>
                            <div style="position: absolute; left: ${entryPercent}%; bottom: 6px; font-size: 14px; color: #ffff88; transform: translateX(-50%); font-weight: bold; text-shadow: 0 0 12px rgba(255,255,0,0.9);">▲</div>
                            
                            <!-- TP1 - LARGER -->
                            <div style="position: absolute; left: ${tp1Percent}%; top: 0; height: 100%; width: 3px; background: linear-gradient(to bottom, ${pos.tp1_hit ? '#00ff88' : '#ffcc00'}, ${pos.tp1_hit ? '#00cc66' : '#ff9900'}, rgba(255,170,0,0.3)); z-index: 4; box-shadow: 0 0 12px ${pos.tp1_hit ? 'rgba(0,255,136,0.9)' : 'rgba(255,170,0,0.8)'};"></div>
                            <div style="position: absolute; left: ${tp1Percent}%; top: 6px; font-size: 14px; color: ${pos.tp1_hit ? '#00ff88' : '#ffcc44'}; transform: translateX(-50%); font-weight: bold; text-shadow: 0 0 12px ${pos.tp1_hit ? 'rgba(0,255,136,0.9)' : 'rgba(255,170,0,0.9)'}; letter-spacing: 1px;">${pos.tp1_hit ? '✓' : 'TP1'}</div>
                            
                            <!-- TP2 - LARGER -->
                            <div style="position: absolute; left: ${tp2Percent}%; top: 0; height: 100%; width: 3px; background: linear-gradient(to bottom, ${pos.tp2_hit ? '#00ff88' : '#00ff99'}, ${pos.tp2_hit ? '#00cc66' : '#00cc77'}, rgba(0,221,136,0.3)); z-index: 4; box-shadow: 0 0 12px ${pos.tp2_hit ? 'rgba(0,255,136,0.9)' : 'rgba(0,255,150,0.8)'};"></div>
                            <div style="position: absolute; left: ${tp2Percent}%; top: 6px; font-size: 14px; color: ${pos.tp2_hit ? '#00ff88' : '#00ffaa'}; transform: translateX(-50%); font-weight: bold; text-shadow: 0 0 12px ${pos.tp2_hit ? 'rgba(0,255,136,0.9)' : 'rgba(0,255,150,0.9)'}; letter-spacing: 1px;">${pos.tp2_hit ? '✓' : 'TP2'}</div>
                            
                            <!-- TP3 - LARGER -->
                            <div style="position: absolute; left: ${tp3Percent}%; top: 0; height: 100%; width: 3px; background: linear-gradient(to bottom, ${pos.tp3_hit ? '#00ff88' : '#00ffff'}, ${pos.tp3_hit ? '#00cc66' : '#00cccc'}, rgba(0,255,255,0.3)); z-index: 4; box-shadow: 0 0 12px ${pos.tp3_hit ? 'rgba(0,255,136,0.9)' : 'rgba(0,255,255,0.8)'};"></div>
                            <div style="position: absolute; left: ${tp3Percent}%; top: 6px; font-size: 14px; color: ${pos.tp3_hit ? '#00ff88' : '#00ffff'}; transform: translateX(-50%); font-weight: bold; text-shadow: 0 0 12px ${pos.tp3_hit ? 'rgba(0,255,136,0.9)' : 'rgba(0,255,255,0.9)'}; letter-spacing: 1px;">${pos.tp3_hit ? '✓' : 'TP3'}</div>
                            
                            <!-- Current Price Bubble - Enhanced -->
                            <div style="position: absolute; left: ${currentPercent}%; top: 50%; transform: translate(-50%, -50%); width: 34px; height: 34px; border-radius: 50%; background: transparent; z-index: 9; animation: ringPulse 2s ease-out infinite; border: 3px solid ${isProfit ? 'rgba(0,255,136,0.5)' : 'rgba(255,68,68,0.5)'};"></div>
                            <div style="position: absolute; left: ${currentPercent}%; top: 50%; transform: translate(-50%, -50%); width: 20px; height: 20px; border-radius: 50%; background: ${isProfit ? 'radial-gradient(circle at 30% 30%, #44ffaa, #00cc66)' : 'radial-gradient(circle at 30% 30%, #ff8888, #dd3333)'}; z-index: 10; animation: scalePulse 1.5s ease-in-out infinite; box-shadow: 0 0 25px ${isProfit ? 'rgba(0,255,136,1)' : 'rgba(255,68,68,1)'}, 0 0 50px ${isProfit ? 'rgba(0,255,136,0.5)' : 'rgba(255,68,68,0.5)'};"></div>
                            <div style="position: absolute; left: ${currentPercent}%; bottom: 2px; font-size: 13px; color: ${isProfit ? '#44ffaa' : '#ff6666'}; transform: translateX(-50%); font-weight: bold; text-shadow: 0 0 12px rgba(0,0,0,1), 0 0 20px ${isProfit ? 'rgba(0,255,136,0.8)' : 'rgba(255,68,68,0.8)'};">$${current.toFixed(2)}</div>
                        </div>
                        
                        <!-- Info Grid - Vibrant Glassmorphism -->
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; font-size: 11px; margin-top: 10px;">
                            <div style="text-align: center; padding: 10px 6px; background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)); backdrop-filter: blur(5px); border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                                <div style="color: #888; font-size: 10px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px;">ENTRY</div>
                                <div style="color: #fff; font-weight: bold; font-size: 12px; text-shadow: 0 0 10px rgba(255,255,255,0.3);">$${(pos.entry || 0).toFixed(2)}</div>
                            </div>
                            <div style="text-align: center; padding: 10px 6px; background: linear-gradient(135deg, rgba(255,68,68,0.15), rgba(255,68,68,0.05)); backdrop-filter: blur(5px); border-radius: 10px; border: 1px solid rgba(255,68,68,0.2);">
                                <div style="color: #ff8888; font-size: 10px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px;">SL</div>
                                <div style="color: #ff6666; font-weight: bold; font-size: 12px; text-shadow: 0 0 10px rgba(255,68,68,0.5);">$${(pos.stop_loss || 0).toFixed(2)}</div>
                            </div>
                            <div style="text-align: center; padding: 10px 6px; background: linear-gradient(135deg, rgba(255,170,0,0.15), rgba(255,170,0,0.05)); backdrop-filter: blur(5px); border-radius: 10px; border: 1px solid rgba(255,170,0,0.2);">
                                <div style="color: #ffcc66; font-size: 10px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px;">TP1</div>
                                <div style="color: ${pos.tp1_hit ? '#00ff88' : '#ffcc44'}; font-weight: bold; font-size: 12px; text-shadow: 0 0 10px ${pos.tp1_hit ? 'rgba(0,255,136,0.5)' : 'rgba(255,170,0,0.5)'};">${pos.tp1_hit ? '✓ HIT' : '$' + (pos.tp1 || 0).toFixed(2)}</div>
                            </div>
                            <div style="text-align: center; padding: 10px 6px; background: linear-gradient(135deg, rgba(0,255,255,0.15), rgba(0,255,255,0.05)); backdrop-filter: blur(5px); border-radius: 10px; border: 1px solid rgba(0,255,255,0.2);">
                                <div style="color: #88ffff; font-size: 10px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px;">TP3</div>
                                <div style="color: ${pos.tp3_hit ? '#00ff88' : '#00ffff'}; font-weight: bold; font-size: 12px; text-shadow: 0 0 10px ${pos.tp3_hit ? 'rgba(0,255,136,0.5)' : 'rgba(0,255,255,0.5)'};">${pos.tp3_hit ? '✓ HIT' : '$' + (pos.tp3 || 0).toFixed(2)}</div>
                            </div>
                        </div>
                    `;
                    
                    if (contentEl) {
                        contentEl.innerHTML = slotHTML;
                    }
                }
                
                // Render Position 1 (first position or empty)
                renderPositionSlot(positions[0] || null, 1);
                
                // Render Position 2 (second position or empty)
                renderPositionSlot(positions[1] || null, 2);
                
                // Re-attach button handlers after rendering
                if (window.setupButtonHandlers) {
                    window.setupButtonHandlers();
                }
                
                // Update Position 2 chart visibility and data
                var pos2ChartContainer = document.getElementById('pos2-chart-container');
                console.log('[P2 Chart Debug] Container found:', !!pos2ChartContainer, 'positions[1]:', positions[1] ? positions[1].symbol : 'none');
                if (pos2ChartContainer) {
                    if (positions[1] && positions[1].symbol) {
                        // Show Position 2 chart
                        console.log('[P2 Chart Debug] Showing chart for', positions[1].symbol);
                        pos2ChartContainer.style.display = 'block';
                        
                        // Update header info
                        var pos2 = positions[1];
                        var pos2SymbolEl = document.getElementById('pos2-chart-symbol');
                        var pos2PriceEl = document.getElementById('pos2-price-display');
                        var pos2PnlEl = document.getElementById('pos2-pnl-display');
                        var pos2EntryEl = document.getElementById('pos2-chart-entry');
                        var pos2SlEl = document.getElementById('pos2-chart-sl');
                        var pos2Tp1El = document.getElementById('pos2-chart-tp1');
                        
                        if (pos2SymbolEl) {
                            var sym2 = pos2.symbol;
                            if (sym2.indexOf('/') === -1 && sym2.endsWith('USDT')) {
                                sym2 = sym2.replace('USDT', '/USDT');
                            }
                            pos2SymbolEl.textContent = sym2;
                        }
                        if (pos2PriceEl) pos2PriceEl.textContent = '$' + (pos2.current_price || pos2.entry || 0).toFixed(4);
                        if (pos2PnlEl) {
                            var pnl2 = pos2.pnl || 0;
                            pos2PnlEl.textContent = (pnl2 >= 0 ? '+$' : '-$') + Math.abs(pnl2).toFixed(2);
                            pos2PnlEl.style.color = pnl2 >= 0 ? '#00ff88' : '#ff4444';
                        }
                        if (pos2EntryEl) pos2EntryEl.textContent = '$' + (pos2.entry || 0).toFixed(4);
                        if (pos2SlEl) pos2SlEl.textContent = '$' + (pos2.stop_loss || 0).toFixed(4);
                        if (pos2Tp1El) pos2Tp1El.textContent = '$' + (pos2.tp1 || 0).toFixed(4);
                        
                        // FIXED: Always update window.pos2Data so entry lines stay current
                        window.pos2Data = pos2;
                        
                        // Fetch Position 2 OHLC data and update chart (throttle to every 3 seconds for responsive updates)
                        if (!window.pos2LastFetch || Date.now() - window.pos2LastFetch > 3000) {
                            window.pos2LastFetch = Date.now();
                            fetchPos2ChartData(pos2.symbol, pos2);
                        } else if (pos2Chart) {
                            // Even when not fetching new OHLC, redraw the chart to update entry lines
                            pos2Chart.update('none');
                        }
                    } else {
                        // Hide Position 2 chart when no second position
                        pos2ChartContainer.style.display = 'none';
                        window.pos2Data = null;  // Clear position data when no position
                    }
                }
                
                // Technical Indicators
                var ind = data.indicators || {};
                setText('ind-rsi', ind.rsi ? ind.rsi.toFixed(1) : '--');
                if (ind.rsi) {
                    var rsiBar = document.getElementById('rsi-bar');
                    if (rsiBar) {
                        rsiBar.style.width = ind.rsi + '%';
                        rsiBar.style.background = ind.rsi > 70 ? '#ff4444' : ind.rsi < 30 ? '#00ff88' : '#00d4ff';
                    }
                }
                setText('ind-macd', ind.macd_signal || '--');
                setText('ind-adx', ind.adx ? ind.adx.toFixed(1) : '--');
                setText('ind-atr', ind.atr ? '$' + ind.atr.toFixed(4) : '--');
                setText('ind-bb', ind.bb_position || '--');
                setText('ind-volume', ind.volume_ratio ? ind.volume_ratio.toFixed(2) + 'x' : '--');
                
                // Market Regime
                var regime = data.regime || {};
                var regimeBadge = document.getElementById('regime-badge');
                if (regimeBadge) {
                    regimeBadge.textContent = regime.regime || 'UNKNOWN';
                    regimeBadge.className = 'badge ' + (regime.regime === 'TRENDING' || regime.regime === 'STRONG_TRENDING' ? 'badge-success' : regime.regime === 'CHOPPY' ? 'badge-danger' : 'badge-info');
                }
                setText('regime-adx', regime.adx ? regime.adx.toFixed(1) : '--');
                setText('regime-hurst', regime.hurst ? regime.hurst.toFixed(3) : '--');
                setText('regime-volatility', regime.volatility || '--');
                var tradeableEl = document.getElementById('regime-tradeable');
                if (tradeableEl) {
                    tradeableEl.textContent = regime.tradeable ? '✅ YES' : '❌ NO';
                    tradeableEl.className = 'indicator-value ' + (regime.tradeable ? 'positive' : 'negative');
                }
                
                // Risk Manager
                var risk = data.risk || {};
                var canTradeEl = document.getElementById('risk-can-trade');
                if (canTradeEl) {
                    canTradeEl.textContent = risk.can_trade ? '✅ YES' : '🛑 NO';
                    canTradeEl.className = 'indicator-value ' + (risk.can_trade ? 'positive' : 'negative');
                }
                setText('risk-mode', risk.dd_mode || 'NORMAL');
                setText('risk-base', ((risk.base_risk || 0.02) * 100).toFixed(1) + '%');
                setText('risk-adjusted', ((risk.adjusted_risk || 0.02) * 100).toFixed(2) + '%');
                setText('risk-kelly', ((risk.kelly_risk || 0.02) * 100).toFixed(2) + '%');
                var dailyPnl = risk.daily_pnl || 0;
                var dailyEl = document.getElementById('risk-daily');
                if (dailyEl) {
                    dailyEl.textContent = (dailyPnl >= 0 ? '+$' : '-$') + Math.abs(dailyPnl).toFixed(2);
                    dailyEl.className = 'indicator-value ' + (dailyPnl >= 0 ? 'positive' : 'negative');
                }
                
                // MTF Analysis
                var mtf = data.mtf || {};
                var mtfPrimary = mtf.primary || {};
                var mtfSecondary = mtf.secondary || {};
                var mtfHigher = mtf.higher || {};
                updateMTF('mtf-3m', (mtfPrimary.trend && mtfPrimary.trend.direction) || '--');
                updateMTF('mtf-15m', (mtfSecondary.trend && mtfSecondary.trend.direction) || '--');
                updateMTF('mtf-1h', (mtfHigher.trend && mtfHigher.trend.direction) || '--');
                setText('mtf-confluence', (mtf.confluence_pct || 0) + '%');
                setText('mtf-alignment', (mtf.alignment_score || 0).toFixed(2));
                setText('mtf-recommendation', mtf.recommendation || '--');
                
                // AI Filter
                var ai = data.ai || {};
                setText('ai-mode', ai.mode || 'filter');
                setText('ai-threshold', ((ai.threshold || 0.7) * 100).toFixed(0) + '%');
                setText('ai-approved', ai.approved || 0);
                setText('ai-rejected', ai.rejected || 0);
                var total = (ai.approved || 0) + (ai.rejected || 0);
                setText('ai-rate', total > 0 ? ((ai.approved || 0) / total * 100).toFixed(0) + '%' : '--');
                setText('ai-last', ai.last_decision || '--');
                
                // ML Model Status
                var ml = data.ml || {};
                var mlBadge = document.getElementById('ml-status-badge');
                if (mlBadge) {
                    if (ml.loaded) {
                        mlBadge.textContent = 'ACTIVE';
                        mlBadge.className = 'badge badge-success';
                        setText('ml-status', ml.status || 'Loaded');
                        setText('ml-accuracy', ((ml.accuracy || 0) * 100).toFixed(1) + '%');
                        setText('ml-samples', ml.samples || 0);
                        setText('ml-features', (ml.features || 24) + ' features');
                        if (ml.last_prediction) {
                            setText('ml-last-pred', ((ml.last_prediction.probability || 0.5) * 100).toFixed(0) + '% win');
                        }
                    } else {
                        mlBadge.textContent = 'NOT LOADED';
                        mlBadge.className = 'badge badge-neutral';
                        setText('ml-status', 'Not Available');
                        setText('ml-accuracy', '--');
                        setText('ml-samples', '--');
                        setText('ml-features', '--');
                        setText('ml-last-pred', '--');
                    }
                }
                
                // AI Decision Tracker
                var tracker = data.ai_tracker || {};
                setText('tracker-total', tracker.total_tracked || 0);
                setText('tracker-approval-rate', tracker.approval_rate || '--');
                var accEl = document.getElementById('tracker-approval-accuracy');
                if (accEl) {
                    var accVal = tracker.approval_accuracy || 'N/A';
                    accEl.textContent = accVal;
                    if (accVal !== 'N/A') {
                        var accNum = parseFloat(accVal);
                        if (accNum >= 60) accEl.className = 'indicator-value positive';
                        else if (accNum >= 50) accEl.className = 'indicator-value warning';
                        else accEl.className = 'indicator-value negative';
                    }
                }
                var netEl = document.getElementById('tracker-net-value');
                if (netEl) {
                    var netVal = tracker.net_ai_value || '$0.00';
                    netEl.textContent = netVal;
                    netEl.className = 'indicator-value ' + (netVal.includes('+') ? 'positive' : netVal.includes('-') ? 'negative' : '');
                }
                
                // Pre-Filter Statistics
                var pf = data.prefilter || {};
                setText('prefilter-total', (pf.raw_signals || 0) + ' raw / ' + (pf.total_signals || 0) + ' filtered');
                setText('prefilter-passed', pf.passed || 0);
                setText('prefilter-score', pf.blocked_by_score || 0);
                setText('prefilter-adx-low', pf.blocked_by_adx_low || 0);
                setText('prefilter-adx-danger', pf.blocked_by_adx_danger || 0);
                setText('prefilter-volume', pf.blocked_by_volume || 0);
                setText('prefilter-confluence', pf.blocked_by_confluence || 0);
                setText('prefilter-btc', pf.blocked_by_btc_filter || 0);
                var pfBadge = document.getElementById('prefilter-badge');
                if (pfBadge) {
                    var passRate = pf.pass_rate || '0%';
                    pfBadge.textContent = passRate + ' PASS';
                    var passNum = parseFloat(passRate);
                    if (passNum >= 70) pfBadge.className = 'badge badge-success';
                    else if (passNum >= 40) pfBadge.className = 'badge badge-warning';
                    else pfBadge.className = 'badge badge-danger';
                }
                
                // Position display is now handled in the enhanced live position view above
            
                // Trading Parameters
                var params = data.params || {};
                setText('param-risk', ((params.risk_pct || 0.02) * 100).toFixed(1) + '%');
                setText('param-atr', (params.atr_mult || 1.5) + 'x');
                setText('param-tp', (params.tp1_r || 1.5) + 'R / ' + (params.tp2_r || 2.5) + 'R / ' + (params.tp3_r || 4) + 'R');
                setText('param-sizing', ((params.tp1_pct || 0.4) * 100) + '% / ' + ((params.tp2_pct || 0.35) * 100) + '% / ' + ((params.tp3_pct || 0.25) * 100) + '%');
                setText('param-trail', (params.trail_trigger_r || 1.5) + 'R');
                
                // Signals Table
                var signalsBody = document.getElementById('signals-body');
                var signals = data.signals || [];
                if (signalsBody) {
                    if (signals.length === 0) {
                        signalsBody.innerHTML = '<tr><td colspan="6" style="color: #666;">No signals yet</td></tr>';
                    } else {
                        var signalsHtml = '';
                        signals.slice(-10).reverse().forEach(function(sig) {
                            var dirClass = sig.direction === 'LONG' ? 'positive' : sig.direction === 'SHORT' ? 'negative' : '';
                            var statusBadge = sig.executed ? '<span class="badge badge-success">EXECUTED</span>' : 
                                               sig.rejected ? '<span class="badge badge-danger">REJECTED</span>' : 
                                               '<span class="badge badge-neutral">PENDING</span>';
                            signalsHtml += '<tr class="trade-row">' +
                                '<td>' + (sig.time || 'N/A') + '</td>' +
                                '<td class="' + dirClass + '">' + (sig.direction || 'N/A') + '</td>' +
                                '<td>' + ((sig.confidence || 0) * 100).toFixed(0) + '%</td>' +
                                '<td>$' + (sig.entry || 0).toFixed(4) + '</td>' +
                                '<td>' + (sig.ai_decision || '--') + '</td>' +
                                '<td>' + statusBadge + '</td>' +
                                '</tr>';
                        });
                        signalsBody.innerHTML = signalsHtml;
                    }
                }
                
                // Trades Table
                var tradesBody = document.getElementById('trades-body');
                var trades = data.trades || [];
                if (tradesBody) {
                    if (trades.length === 0) {
                        tradesBody.innerHTML = '<tr><td colspan="8" style="color: #666;">No trades yet</td></tr>';
                    } else {
                        var tradesHtml = '';
                        trades.slice(-10).reverse().forEach(function(trade) {
                            // Use trade.result field if available, otherwise determine from PnL
                            var isWin = trade.result ? (trade.result.toUpperCase() === 'WIN') : ((trade.pnl || 0) >= 0);
                            var pnlClass = (trade.pnl || 0) >= 0 ? 'positive' : 'negative';
                            var resultBadge = isWin ? 
                                '<span class="badge badge-success">WIN</span>' : 
                                '<span class="badge badge-danger">LOSS</span>';
                            
                            // Position number - default to 1, use 2 for secondary positions
                            var posNum = trade.position_num || 1;
                            var posBadge = posNum === 2 ? 
                                '<span style="background:#b366ff22;color:#b366ff;padding:2px 6px;border-radius:4px;font-size:10px;">P2</span>' :
                                '<span style="background:#00d4ff22;color:#00d4ff;padding:2px 6px;border-radius:4px;font-size:10px;">P1</span>';
                            
                            // Extract date and times from closed_at if explicit fields are missing
                            var tradeDate = trade.date;
                            var entryTime = trade.entry_time;
                            var exitTime = trade.exit_time || trade.time;
                            
                            // Parse from closed_at if fields are null/undefined
                            if ((!tradeDate || tradeDate === 'null') && trade.closed_at) {
                                try {
                                    var closedDate = new Date(trade.closed_at);
                                    tradeDate = closedDate.toISOString().split('T')[0];  // YYYY-MM-DD
                                    if (!exitTime || exitTime === 'null') {
                                        exitTime = closedDate.toTimeString().split(' ')[0];  // HH:MM:SS
                                    }
                                } catch(e) {}
                            }
                            
                            // Default to N/A if still missing
                            tradeDate = tradeDate || 'N/A';
                            entryTime = entryTime || 'N/A';
                            exitTime = exitTime || 'N/A';
                            
                            var symbol = (trade.symbol || 'N/A').replace('USDT', '');
                            var sideBadge = trade.side === 'LONG' ? 
                                '<span style="color:#00ff88;">▲ LONG</span>' :
                                '<span style="color:#ff4444;">▼ SHORT</span>';
                            tradesHtml += '<tr class="trade-row">' +
                                '<td style="font-size:11px;color:#888;">' + tradeDate + '</td>' +
                                '<td>' + posBadge + '</td>' +
                                '<td style="font-weight:bold;">' + symbol + '</td>' +
                                '<td>' + sideBadge + '</td>' +
                                '<td style="font-size:11px;">$' + (trade.entry || 0).toFixed(4) + ' → $' + (trade.exit || 0).toFixed(4) + '</td>' +
                                '<td style="font-size:11px;color:#888;">' + entryTime + ' → ' + exitTime + '</td>' +
                                '<td class="' + pnlClass + '">' + (trade.pnl >= 0 ? '+$' : '-$') + Math.abs(trade.pnl || 0).toFixed(2) + '</td>' +
                                '<td>' + resultBadge + '</td>' +
                                '</tr>';
                        });
                        tradesBody.innerHTML = tradesHtml;
                    }
                }
                
                // Equity Chart
                updateEquityChart(data.equity_curve || []);
            } catch (err) {
                console.error('ERROR in updateDashboard:', err.message, err.stack);
            }
            
            // Update pipeline monitor
            updatePipelineMonitor();
        }
        
        // ============================================================
        // PIPELINE MONITOR - Ship's Engine Room Visualization
        // ============================================================
        var lastPipelineData = null;
        
        async function updatePipelineMonitor() {
            try {
                const response = await fetch('/api/pipeline');
                const data = await response.json();
                lastPipelineData = data;
                renderPipeline(data);
            } catch (err) {
                console.error('Pipeline fetch error:', err);
            }
        }
        
        var currentOpenPopup = null;  // Track which popup is open
        
        function renderPipeline(data) {
            const overallBadge = document.getElementById('pipeline-overall-status');
            const alertsDiv = document.getElementById('pipeline-alerts');
            const alertList = document.getElementById('pipeline-alert-list');
            
            // Update overall status badge
            if (overallBadge) {
                const status = data.overall_status || 'unknown';
                overallBadge.textContent = status.toUpperCase();
                overallBadge.className = 'badge ' + 
                    (status === 'ok' ? 'badge-success' : 
                     status === 'warning' ? 'badge-warning' : 
                     status === 'error' ? 'badge-danger' : 'badge-neutral');
            }
            
            // Update engine status bar
            const d = window.lastData || {};
            const engineCycle = document.getElementById('engine-cycle');
            const engineSignal = document.getElementById('engine-signal');
            const engineRisk = document.getElementById('engine-risk');
            const engineAi = document.getElementById('engine-ai');
            const engineUptime = document.getElementById('engine-uptime');
            
            // Get engine component for accurate data
            const engineComp = data.components && data.components.engine;
            if (engineCycle) {
                if (engineComp && engineComp.cycle_count) {
                    engineCycle.textContent = engineComp.cycle_count;
                } else {
                    engineCycle.textContent = data.cycle_count || d.cycle_count || '--';
                }
            }
            if (engineSignal) {
                const sig = d.signal_direction || d.last_signal || '--';
                engineSignal.textContent = sig.toUpperCase();
                engineSignal.style.color = sig === 'long' ? '#00ff88' : sig === 'short' ? '#ff4444' : '#888';
            }
            if (engineRisk) {
                const comp = data.components && data.components.risk_manager;
                engineRisk.textContent = comp && comp.can_trade ? 'OK' : 'BLOCKED';
                engineRisk.style.color = comp && comp.can_trade ? '#00ff88' : '#ff4444';
            }
            if (engineAi) {
                const comp = data.components && data.components.ai_filter;
                if (comp && comp.in_cooldown) {
                    engineAi.textContent = 'CD ' + Math.round(comp.cooldown_remaining || 0) + 's';
                    engineAi.style.color = '#aa00ff';
                } else {
                    engineAi.textContent = comp ? (comp.mode || 'READY').toUpperCase() : '--';
                    engineAi.style.color = '#00d4ff';
                }
            }
            if (engineUptime) {
                // Use engine component uptime if available
                const uptime = (engineComp && engineComp.uptime_seconds) || d.uptime;
                if (uptime) {
                    const hrs = Math.floor(uptime / 3600);
                    const mins = Math.floor((uptime % 3600) / 60);
                    engineUptime.textContent = hrs + 'h ' + mins + 'm';
                } else {
                    engineUptime.textContent = '--';
                }
            }
            
            // Update component status bar (the new unified status bar)
            const components = data.components || {};
            const compMap = {
                engine: { id: 'comp-engine', name: 'ENGINE' },
                market_feed: { id: 'comp-market', name: 'MARKET' },
                indicators: { id: 'comp-indicators', name: 'IND' },
                regime: { id: 'comp-regime', name: 'REG' },
                ai_filter: { id: 'comp-ai', name: 'AI' },
                risk_manager: { id: 'comp-risk', name: 'RISK' },
                position_monitor: { id: 'comp-position', name: 'POS' },
                exchange: { id: 'comp-exchange', name: 'EXCH' },
                telegram: { id: 'comp-telegram', name: 'TELE' }
            };
            
            // Get main dashboard data as fallback
            const mainData = window.lastDashboardData || window.lastData || {};
            const indicators = mainData.indicators || {};
            const regime = mainData.regime || {};
            const ai = mainData.ai || {};
            const risk = mainData.risk || {};
            const status = mainData.status || {};
            const position = mainData.open_position || {};
            
            Object.keys(compMap).forEach(key => {
                const def = compMap[key];
                let comp = components[key] || { status: 'unknown' };
                const el = document.getElementById(def.id);
                if (!el) return;
                
                // Enrich with main dashboard data if pipeline data is missing
                if (key === 'engine') {
                    // Engine status from pipeline - no fallback, must be real
                    // If no engine data and we have other data, engine is stopped
                    if (!comp.running && Object.keys(components).length > 0) {
                        comp.status = 'stopped';
                    }
                }
                // Use fallback data ONLY when pipeline data is missing (don't overwrite valid data)
                if (key === 'market_feed') {
                    // Prefer pipeline data, fallback to main dashboard data
                    if (!comp.last_price && status.price) comp.last_price = status.price;
                    if (!comp.symbol && status.symbol) comp.symbol = status.symbol;
                    if (!comp.status || comp.status === 'unknown') comp.status = status.connected ? 'ok' : 'error';
                }
                if (key === 'indicators') {
                    if (!comp.rsi && indicators.rsi !== undefined) comp.rsi = indicators.rsi;
                    if (!comp.adx && indicators.adx !== undefined) comp.adx = indicators.adx;
                    if (comp.rsi && (!comp.status || comp.status === 'unknown')) comp.status = 'ok';
                }
                if (key === 'regime') {
                    if (!comp.current && regime.regime) comp.current = regime.regime;
                }
                if (key === 'ai_filter') {
                    if (ai.total_signals !== undefined && !comp.call_count) comp.call_count = ai.total_signals;
                }
                if (key === 'risk_manager' && risk) {
                    if (comp.can_trade === undefined) comp.can_trade = !risk.in_cooldown;
                    if (risk.in_cooldown && (!comp.status || comp.status === 'ok')) comp.status = 'warning';
                }
                if (key === 'position_monitor') {
                    if (comp.has_position === undefined) {
                        const hasPos = position && position.side;
                        comp.has_position = hasPos;
                        comp.position_side = position.side;
                    }
                }
                if (key === 'exchange' && status.connected !== undefined) {
                    comp = { ...comp, connected: status.connected, status: status.connected ? 'ok' : 'error' };
                }
                // Telegram - use enhanced pipeline data
                if (key === 'telegram') {
                    // Don't override with fallback - use actual status
                }
                
                const compStatus = comp.status || 'unknown';
                const span = el.querySelector('span');
                
                // Get short status text with enhanced data
                let text = '--';
                if (key === 'engine') {
                    if (!comp.running) text = 'OFF';
                    else if (comp.paused) text = 'PAUSE';
                    else text = '#' + (comp.cycle_count || 0);
                }
                else if (key === 'market_feed') text = comp.last_price ? '$' + comp.last_price.toFixed(0) : (comp.status === 'ok' ? 'OK' : '--');
                else if (key === 'indicators') text = comp.rsi ? 'RSI ' + Math.round(comp.rsi) : 'OK';
                else if (key === 'regime') text = (comp.current || '--').substring(0, 4).toUpperCase();
                else if (key === 'ai_filter') text = comp.in_cooldown ? 'CD' : 'RDY';
                else if (key === 'risk_manager') text = comp.can_trade ? '✓' : '✗';
                else if (key === 'position_monitor') text = comp.has_position ? 'ACT' : 'NIL';
                else if (key === 'exchange') text = comp.connected ? '✓' : '✗';
                else if (key === 'telegram') {
                    if (comp.status === 'disabled') text = 'OFF';
                    else if (comp.status === 'initializing') text = '...';
                    else if (comp.status === 'error') text = 'ERR';
                    else if (comp.status === 'warning') text = '⚠';
                    else if (comp.is_started) text = '✓';
                    else text = '✗';
                }
                
                if (span) span.textContent = text;
                
                // Set status class
                el.className = 'comp-status';
                if (key === 'engine') el.classList.add('engine-status');
                
                if (compStatus === 'ok' || compStatus === 'active') el.classList.add('ok');
                else if (compStatus === 'warning' || compStatus === 'cooldown' || compStatus === 'idle') el.classList.add('warning');
                else if (compStatus === 'error' || comp.error) el.classList.add('error');
                else if (compStatus === 'disabled' || compStatus === 'offline') el.classList.add('disabled');
                else if (compStatus === 'initializing') el.classList.add('initializing');
                else if (compStatus === 'stopped' || (!comp.running && key === 'engine')) el.classList.add('stopped');
                
                // Store data for popup
                el.dataset.key = key;
                el.onclick = function() { togglePipelinePopup(key, el); };
            });
            
            // Show alerts
            const allAlerts = [...(data.errors || []), ...(data.warnings || [])];
            if (allAlerts.length > 0 && alertsDiv && alertList) {
                alertsDiv.style.display = 'block';
                alertList.innerHTML = allAlerts.map(a => '<div style="color:#ff4444;">• ' + a + '</div>').join('');
            } else if (alertsDiv) {
                alertsDiv.style.display = 'none';
            }
        }
        
        function togglePipelinePopup(key, element) {
            // Close existing popup if clicking same component
            const existingPopup = document.querySelector('.pipeline-popup.active');
            if (existingPopup) {
                const existingKey = existingPopup.getAttribute('data-key');
                existingPopup.remove();
                if (existingKey === key) {
                    currentOpenPopup = null;
                    return; // Toggle off
                }
            }
            
            // Get component data - use pipeline data or fallback to main dashboard data
            const pipelineComps = (lastPipelineData && lastPipelineData.components) ? lastPipelineData.components : {};
            const d = window.lastData || window.lastDashboardData || {};
            
            // Build component data from available sources
            let comp = pipelineComps[key] || {};
            
            // Fallback enrichment from main dashboard data
            if (key === 'indicators' && d.indicators) {
                comp = { ...comp, ...d.indicators, status: 'ok' };
            } else if (key === 'regime' && d.regime) {
                comp = { ...comp, current: d.regime.regime, ...d.regime, status: 'ok' };
            } else if (key === 'ai_filter' && d.ai) {
                comp = { ...comp, ...d.ai, status: 'ok' };
            } else if (key === 'risk_manager' && d.risk) {
                comp = { ...comp, ...d.risk, status: 'ok' };
            } else if (key === 'market_feed' && d.status) {
                comp = { ...comp, symbol: d.status.symbol, last_price: d.status.price, status: d.status.connected ? 'ok' : 'error' };
            } else if (key === 'position_monitor' && d.open_position) {
                comp = { ...comp, has_position: !!d.open_position.side, position_side: d.open_position.side, status: 'ok' };
            }
            
            // Build popup content
            let html = '<div class="pipeline-popup-title">' + getComponentIcon(key) + ' ' + getComponentName(key) + '</div>';
            html += buildPopupContent(key, comp, d);
            
            // Create popup element
            const popup = document.createElement('div');
            popup.className = 'pipeline-popup active';
            popup.setAttribute('data-key', key);
            popup.innerHTML = html;
            
            // Position relative to clicked element
            element.style.position = 'relative';
            element.appendChild(popup);
            currentOpenPopup = key;
        }
        
        function getComponentIcon(key) {
            const icons = { market_feed: '📡', indicators: '📊', regime: '🎯', ai_filter: '🤖', risk_manager: '🛡️', position_monitor: '💼', exchange: '💱', telegram: '📱' };
            return icons[key] || '⚙️';
        }
        
        function getComponentName(key) {
            const names = { engine: 'Trading Engine', market_feed: 'Market Feed', indicators: 'Indicators', regime: 'Market Regime', ai_filter: 'AI Filter', risk_manager: 'Risk Manager', position_monitor: 'Position Monitor', exchange: 'Exchange', telegram: 'Telegram Bot' };
            return names[key] || key;
        }
        
        function buildPopupContent(key, comp, d) {
            let html = '';
            const row = (label, value, cls) => '<div class="pipeline-popup-row"><span class="pipeline-popup-label">' + label + '</span><span class="pipeline-popup-value ' + (cls || '') + '">' + value + '</span></div>';
            
            // Status row always first
            const statusCls = comp.status === 'ok' ? 'ok' : comp.status === 'error' || comp.status === 'stopped' ? 'error' : 'warning';
            html += row('Status', (comp.status || 'unknown').toUpperCase(), statusCls);
            if (comp.error) html += row('Error', comp.error, 'error');
            
            switch(key) {
                case 'engine':
                    html += row('Running', comp.running ? 'YES' : 'NO', comp.running ? 'ok' : 'error');
                    html += row('Paused', comp.paused ? 'YES' : 'NO', comp.paused ? 'warning' : 'ok');
                    html += row('Cycles', comp.cycle_count || 0);
                    if (comp.last_cycle_seconds_ago != null) {
                        html += row('Last Cycle', comp.last_cycle_seconds_ago.toFixed(0) + 's ago', comp.last_cycle_seconds_ago > 30 ? 'warning' : 'ok');
                    }
                    if (comp.last_cycle_duration_ms) {
                        html += row('Cycle Time', comp.last_cycle_duration_ms.toFixed(0) + 'ms');
                    }
                    if (comp.uptime_seconds) {
                        const hrs = Math.floor(comp.uptime_seconds / 3600);
                        const mins = Math.floor((comp.uptime_seconds % 3600) / 60);
                        html += row('Uptime', hrs + 'h ' + mins + 'm');
                    }
                    break;
                case 'market_feed':
                    html += row('Symbol', comp.symbol || d.symbol || '--', 'info');
                    html += row('Price', comp.last_price ? '$' + comp.last_price.toFixed(4) : '--');
                    html += row('Bars', comp.bars_loaded || '--');
                    html += row('Age', comp.data_age_seconds ? comp.data_age_seconds.toFixed(0) + 's' : '--', comp.data_age_seconds > 300 ? 'warning' : 'ok');
                    break;
                case 'indicators':
                    html += row('RSI', comp.rsi ? comp.rsi.toFixed(1) : (d.rsi ? d.rsi.toFixed(1) : '--'), comp.rsi > 70 ? 'error' : comp.rsi < 30 ? 'ok' : '');
                    html += row('ADX', comp.adx ? comp.adx.toFixed(1) : (d.adx ? d.adx.toFixed(1) : '--'), (comp.adx || d.adx) > 25 ? 'ok' : 'warning');
                    html += row('ATR', comp.atr ? comp.atr.toFixed(6) : '--');
                    if (d.sma_fast) html += row('SMA 15', '$' + d.sma_fast.toFixed(2));
                    if (d.sma_slow) html += row('SMA 40', '$' + d.sma_slow.toFixed(2));
                    break;
                case 'regime':
                    html += row('Regime', (comp.current || d.regime || '--').toUpperCase(), 'info');
                    html += row('Tradeable', comp.tradeable ? 'YES' : 'NO', comp.tradeable ? 'ok' : 'error');
                    html += row('Hurst', comp.hurst ? comp.hurst.toFixed(3) : '--');
                    break;
                case 'ai_filter':
                    html += row('Mode', (comp.mode || '--').toUpperCase(), 'info');
                    html += row('Available', comp.available ? 'YES' : 'NO', comp.available ? 'ok' : 'error');
                    html += row('Cooldown', comp.in_cooldown ? Math.round(comp.cooldown_remaining) + 's' : 'NO', comp.in_cooldown ? 'warning' : 'ok');
                    html += row('Calls', comp.call_count || 0);
                    break;
                case 'risk_manager':
                    html += row('Can Trade', comp.can_trade ? 'YES' : 'NO', comp.can_trade ? 'ok' : 'error');
                    html += row('Daily Limit', comp.daily_limit_hit ? 'HIT' : 'OK', comp.daily_limit_hit ? 'error' : 'ok');
                    html += row('Daily P&L', '$' + (comp.daily_pnl || 0).toFixed(2), (comp.daily_pnl || 0) >= 0 ? 'ok' : 'error');
                    html += row('Mode', comp.mode || '--');
                    break;
                case 'position_monitor':
                    html += row('Position', comp.has_position ? 'YES' : 'NO', 'info');
                    if (comp.has_position) {
                        html += row('Side', (comp.position_side || '--').toUpperCase(), comp.position_side === 'long' ? 'ok' : 'error');
                        html += row('SL Active', comp.sl_protection_active ? 'YES' : 'NO', comp.sl_protection_active ? 'ok' : 'warning');
                    }
                    break;
                case 'exchange':
                    html += row('Connected', comp.connected ? 'YES' : 'NO', comp.connected ? 'ok' : 'error');
                    html += row('Mode', comp.paper_trading ? 'PAPER' : 'LIVE', comp.paper_trading ? 'warning' : 'ok');
                    html += row('Dry Run', comp.dry_run ? 'YES' : 'NO');
                    break;
                case 'telegram':
                    html += row('Enabled', comp.enabled ? 'YES' : 'NO', comp.enabled ? 'info' : 'warning');
                    html += row('Started', comp.is_started ? 'YES' : 'NO', comp.is_started ? 'ok' : 'warning');
                    html += row('Bot Init', comp.bot_initialized ? 'YES' : 'NO', comp.bot_initialized ? 'ok' : 'error');
                    if (comp.error_count > 0) {
                        html += row('Errors', comp.error_count, 'error');
                    }
                    if (comp.last_error) {
                        const errText = comp.last_error.length > 35 ? comp.last_error.substring(0, 35) + '...' : comp.last_error;
                        html += row('Last Error', errText, 'error');
                    }
                    if (comp.seconds_since_success != null) {
                        html += row('Last OK', comp.seconds_since_success + 's ago', comp.seconds_since_success > 60 ? 'warning' : 'ok');
                    }
                    break;
            }
            return html;
        }
        
        // Close popup when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.pipeline-component') && !e.target.closest('.pipeline-popup')) {
                const popup = document.querySelector('.pipeline-popup.active');
                if (popup) popup.remove();
                currentOpenPopup = null;
            }
        });
        
        function updateMTF(id, direction) {
            var el = document.getElementById(id);
            if (el) {
                el.textContent = (direction || '--').toUpperCase();
                el.className = 'mtf-trend ' + (direction === 'bullish' ? 'bullish' : direction === 'bearish' ? 'bearish' : 'neutral');
            }
        }
        
        function updateEquityChart(equityCurve) {
            try {
                var canvas = document.getElementById('equityChart');
                if (!canvas) return;
                var ctx = canvas.getContext('2d');
                if (!equityCurve || equityCurve.length === 0) return;
                
                if (equityChart) {
                    equityChart.data.labels = equityCurve.map((_, i) => i);
                    equityChart.data.datasets[0].data = equityCurve;
                    equityChart.update();
                } else {
                    equityChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: equityCurve.map((_, i) => i),
                            datasets: [{
                                label: 'Equity',
                                data: equityCurve,
                                borderColor: '#00d4ff',
                                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                                fill: true,
                                tension: 0.4,
                                pointRadius: 0,
                                borderWidth: 2
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: { legend: { display: false } },
                            scales: {
                            x: { display: false },
                            y: {
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: { color: '#666', callback: v => '$' + v.toLocaleString() }
                            }
                        }
                    }
                });
                }
            } catch (err) {
                console.error('Error in updateEquityChart:', err.message);
            }
        }
        
        // Initialize date picker with min/max dates (30 days back, today)
        function initDatePicker() {
            var picker = document.getElementById('chart-date-picker');
            if (picker) {
                var today = new Date();
                var minDate = new Date();
                minDate.setDate(today.getDate() - 30);
                
                picker.max = today.toISOString().split('T')[0];
                picker.min = minDate.toISOString().split('T')[0];
            }
        }
        initDatePicker();
        
        // Initial data fetch
        fetchData();
        fetchPriceData();
        fetchLogs();
        fetchMarketScan();
        fetchErrors();
        fetchNews();
        
        // Auto-refresh intervals
        setInterval(fetchData, 2000);        // Main data every 2 seconds (faster position updates)
        setInterval(fetchPriceData, 3000);   // Price chart every 3 seconds
        setInterval(fetchLogs, 10000);       // Logs every 10 seconds
        setInterval(fetchMarketScan, 15000); // Market scan every 15 seconds
        setInterval(fetchErrors, 10000);     // Error status every 10 seconds
        setInterval(fetchNews, 60000);       // News every 60 seconds
        
        // Monitor chart symbol changes (position OR scanner) and force chart refresh
        var lastChartSymbol = null;
        setInterval(function() {
            var currentChartSym = getStableChartSymbol();
            if (currentChartSym && currentChartSym !== lastChartSymbol) {
                console.log('[Dashboard] Chart symbol changed:', lastChartSymbol, '->', currentChartSym, '(source:', window._chartSymbolSource, '). Forcing chart refresh.');
                lastChartSymbol = currentChartSym;
                fetchPriceData(); // Force immediate refresh when symbol changes
            }
        }, 1000);  // Check every 1 second
        
        // Animation loop for entry line glow effect
        function animateEntryLine() {
            if (priceChart && window.lastDashboardData && window.lastDashboardData.open_position) {
                priceChart.update('none'); // Update without animation for smooth glow
            }
            requestAnimationFrame(animateEntryLine);
        }
        // Start animation loop with throttling
        var lastAnimFrame = 0;
        function throttledAnimateEntryLine(timestamp) {
            if (timestamp - lastAnimFrame > 50) { // ~20 FPS for smooth but efficient animation
                lastAnimFrame = timestamp;
                if (priceChart && window.lastDashboardData && window.lastDashboardData.open_position) {
                    priceChart.update('none');
                }
            }
            requestAnimationFrame(throttledAnimateEntryLine);
        }
        requestAnimationFrame(throttledAnimateEntryLine);
        
        // ============================================================
        // THE MACHINE - Trading System Animation (Embedded)
        // ADVANCED VERSION with Neural Network Effects
        // ============================================================
        var machineAnimationId = null;
        var machineInitialized = false;
        
        function initMachineAnimation() {
            if (machineInitialized) return;
            machineInitialized = true;
            
            const canvas = document.getElementById('machine-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            
            // Set canvas size with device pixel ratio for sharpness
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            ctx.scale(dpr, dpr);
            
            const W = rect.width;
            const H = rect.height;
            
            // Get live data
            const data = window.lastDashboardData || {};
            const status = data.status || 'IDLE';
            const signal = data.signal || {};
            const position = data.open_position || null;
            const mlStats = data.ml_stats || {};
            const regime = data.regime || {};
            
            // Animation state
            let time = 0;
            let dataParticles = [];
            let bgParticles = [];
            let pulseWaves = [];
            let neuralSynapses = [];
            
            // Node definitions - Real Julaba Pipeline (left-to-right linear flow)
            // The actual flow: Bybit → OHLCV → Indicators → Regime/Scanner → PhD Math → Filters → Gemini AI → Risk → Execute → Position → Telegram
            const centerY = H / 2;
            const scaleX = W / 1000;
            const topY = centerY - 55;
            const botY = centerY + 55;
            
            const nodes = {
                // STAGE 1: Data Source (left edge)
                bybit: { x: 45*scaleX, y: centerY, label: 'BYBIT', color: '#f7a600', icon: '💱', sub: '--',
                    desc: 'Bybit Exchange API', inputs: ['WebSocket', 'REST API'], outputs: ['OHLCV', 'Ticker', 'OrderBook'] },
                
                // STAGE 2: Data Processing
                ohlcv: { x: 140*scaleX, y: centerY, label: 'OHLCV', color: '#00d4ff', icon: '🕯️', sub: '--',
                    desc: 'Candlestick Data (15m)', inputs: ['Bybit Klines'], outputs: ['Open', 'High', 'Low', 'Close', 'Volume'] },
                
                // STAGE 3: Technical Analysis (split into two paths)
                indicators: { x: 240*scaleX, y: topY, label: 'IND', color: '#ff6600', icon: '📈', sub: '--',
                    desc: 'Technical Indicators', inputs: ['OHLCV'], outputs: ['RSI', 'ADX', 'ATR', 'MACD', 'BB', 'EMA'] },
                regime: { x: 240*scaleX, y: botY, label: 'REG', color: '#ffcc00', icon: '🌊', sub: '--',
                    desc: 'Market Regime (Hurst)', inputs: ['OHLCV', 'Indicators'], outputs: ['TRENDING', 'RANGING', 'VOLATILE'] },
                
                // STAGE 4: Unified Scanner (merges back)
                scanner: { x: 370*scaleX, y: centerY, label: 'SCAN', color: '#aa00ff', icon: '🔬', sub: '--', size: 1.15,
                    desc: 'Unified Scanner (50 pairs)', inputs: ['Indicators', 'Regime'], outputs: ['Top Opportunities', 'Pre-filter'] },
                
                // STAGE 5: PhD Math Scoring (the brain's left hemisphere)
                math: { x: 490*scaleX, y: centerY, label: 'PhD', color: '#00ff88', icon: '🧮', sub: '--', size: 1.3,
                    desc: 'PhD Math Scoring (13 components)', inputs: ['Scanner Candidates'], outputs: ['Score 0-100', 'Confidence', 'Penalties'] },
                
                // STAGE 6: Gemini AI (the brain's right hemisphere)
                gemini: { x: 610*scaleX, y: centerY, label: 'AI', color: '#4285f4', icon: '🤖', sub: '--', size: 1.3,
                    desc: 'Gemini 2.0 Flash AI', inputs: ['Math Score', 'Market Context'], outputs: ['APPROVE', 'REJECT', 'Reasoning'] },
                
                // STAGE 7: Risk & Execution (split)
                risk: { x: 720*scaleX, y: topY, label: 'RISK', color: '#ff4444', icon: '🛡️', sub: '--',
                    desc: 'Risk Manager', inputs: ['AI Approval'], outputs: ['Position Size', 'Leverage', 'SL/TP'] },
                execute: { x: 720*scaleX, y: botY, label: 'EXEC', color: '#ffff00', icon: '⚡', sub: '--',
                    desc: 'Order Execution', inputs: ['Risk Params'], outputs: ['Market Order', 'Server SL/TP'] },
                
                // STAGE 8: Position Management (merge)
                position: { x: 840*scaleX, y: centerY, label: 'POS', color: '#00ff88', icon: '💼', sub: '--',
                    desc: 'Position Monitor', inputs: ['Executed Order'], outputs: ['Trailing Stop', 'TP Hits', 'Exit Signal'] },
                
                // STAGE 9: Output
                telegram: { x: 940*scaleX, y: centerY, label: 'TELE', color: '#0088cc', icon: '📱', sub: '--',
                    desc: 'Telegram Notifications', inputs: ['Trade Events'], outputs: ['Alerts', 'Commands', 'Chat'] }
            };
            
            // Expose nodes globally for live updates
            window.machineNodes = nodes;
            
            // Initialize background circuit particles
            for (let i = 0; i < 30; i++) {
                bgParticles.push({
                    x: Math.random() * W,
                    y: Math.random() * H,
                    vx: (Math.random() - 0.5) * 0.3,
                    vy: (Math.random() - 0.5) * 0.3,
                    size: 1 + Math.random() * 2,
                    alpha: 0.1 + Math.random() * 0.3,
                    color: ['#00d4ff', '#00ff88', '#aa00ff', '#ff6600'][Math.floor(Math.random() * 4)]
                });
            }
            
            // Tooltip element - add to body for no overflow issues
            let tooltip = document.getElementById('machine-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.id = 'machine-tooltip';
                tooltip.style.cssText = 'position:fixed;display:none;background:rgba(10,15,30,0.95);border:2px solid #00d4ff;border-radius:10px;padding:12px 15px;color:#fff;font-size:12px;pointer-events:none;z-index:9999;max-width:280px;box-shadow:0 0 20px rgba(0,212,255,0.4);backdrop-filter:blur(10px);';
                document.body.appendChild(tooltip);
            }
            
            // Track currently shown node for toggle
            let currentNodeKey = null;
            
            // Generate live tooltip content based on node key and live data
            function generateNodeTooltip(key, node) {
                // Use multiple data sources for completeness
                const data = window.lastData || window.lastDashboardData || {};
                const pipelineData = window.lastPipelineData || {};
                const pipelineComps = pipelineData.components || {};
                
                const indicators = data.indicators || {};
                const regime = data.regime || {};
                const signal = data.signal || data.current_signal || {};
                const position = data.open_position || {};
                const ml = data.ml || data.ml_stats || {};
                const risk = data.risk || {};
                const mtf = data.mtf || {};
                const ai = data.ai || {};
                const prefilter = data.prefilter || {};
                const status = data.status || {};
                
                let liveSection = '';
                
                switch(key) {
                    case 'bybit':
                        const marketPrice = (pipelineComps.market_feed && pipelineComps.market_feed.last_price) || status.current_price || '--';
                        const vol24h = status.volume_24h ? (status.volume_24h / 1e6).toFixed(1) + 'M' : '--';
                        const priceStr = marketPrice !== '--' ? '$' + parseFloat(marketPrice).toFixed(4) : '--';
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(247,166,0,0.1);border-radius:6px;border-left:3px solid #f7a600;">
                                <div style="color:#f7a600;font-weight:bold;margin-bottom:6px;">💱 BYBIT EXCHANGE:</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Symbol:</span><span style="color:#fff;">${status.symbol || '--'}</span>
                                    <span style="color:#888;">Price:</span><span style="color:#00ff88;">${priceStr}</span>
                                    <span style="color:#888;">24h Vol:</span><span style="color:#fff;">${vol24h}</span>
                                    <span style="color:#888;">Status:</span><span style="color:${status.connected ? '#00ff88' : '#ff4444'};">${status.connected ? 'CONNECTED' : 'DISCONNECTED'}</span>
                                </div>
                            </div>`;
                        break;
                    case 'ohlcv':
                        const ohlcList = Array.isArray(data.ohlc) ? data.ohlc : [];
                        const lastCandle = ohlcList.length > 0 ? ohlcList[ohlcList.length - 1] : null;
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(0,212,255,0.1);border-radius:6px;border-left:3px solid #00d4ff;">
                                <div style="color:#00d4ff;font-weight:bold;margin-bottom:6px;">🕯️ CANDLE DATA:</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Timeframe:</span><span style="color:#fff;">15m</span>
                                    <span style="color:#888;">Open:</span><span style="color:#fff;">${lastCandle ? '$' + lastCandle.o.toFixed(4) : '--'}</span>
                                    <span style="color:#888;">High:</span><span style="color:#00ff88;">${lastCandle ? '$' + lastCandle.h.toFixed(4) : '--'}</span>
                                    <span style="color:#888;">Low:</span><span style="color:#ff4444;">${lastCandle ? '$' + lastCandle.l.toFixed(4) : '--'}</span>
                                    <span style="color:#888;">Close:</span><span style="color:#fff;">${lastCandle ? '$' + lastCandle.c.toFixed(4) : '--'}</span>
                                    <span style="color:#888;">Volume:</span><span style="color:#fff;">${lastCandle ? lastCandle.v.toFixed(0) : '--'}</span>
                                </div>
                            </div>`;
                        break;
                    case 'indicators':
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(255,102,0,0.1);border-radius:6px;border-left:3px solid #ff6600;">
                                <div style="color:#ff6600;font-weight:bold;margin-bottom:6px;">📊 LIVE VALUES:</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">RSI:</span><span style="color:${(indicators.rsi || 50) > 70 ? '#ff4444' : (indicators.rsi || 50) < 30 ? '#00ff88' : '#fff'};">${(indicators.rsi || 0).toFixed(1)}</span>
                                    <span style="color:#888;">ADX:</span><span style="color:${(indicators.adx || 0) > 25 ? '#00ff88' : '#ffcc00'};">${(indicators.adx || 0).toFixed(1)}</span>
                                    <span style="color:#888;">ATR:</span><span style="color:#fff;">${(indicators.atr || 0).toFixed(4)}</span>
                                    <span style="color:#888;">MACD:</span><span style="color:${(indicators.macd || 0) > 0 ? '#00ff88' : '#ff4444'};">${(indicators.macd || 0).toFixed(4)}</span>
                                    <span style="color:#888;">BB Upper:</span><span style="color:#fff;">${indicators.bb_upper ? '$' + indicators.bb_upper.toFixed(2) : '--'}</span>
                                    <span style="color:#888;">BB Lower:</span><span style="color:#fff;">${indicators.bb_lower ? '$' + indicators.bb_lower.toFixed(2) : '--'}</span>
                                    <span style="color:#888;">EMA 21:</span><span style="color:#fff;">${indicators.ema_21 ? '$' + indicators.ema_21.toFixed(2) : '--'}</span>
                                    <span style="color:#888;">EMA 50:</span><span style="color:#fff;">${indicators.ema_50 ? '$' + indicators.ema_50.toFixed(2) : '--'}</span>
                                </div>
                            </div>`;
                        break;
                    case 'scanner':
                        const scanPairsCount = (pipelineComps.scanner && pipelineComps.scanner.pairs_scanned) || 50;
                        const scanTop = (pipelineComps.scanner && pipelineComps.scanner.top_opportunities) || '--';
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(170,0,255,0.1);border-radius:6px;border-left:3px solid #aa00ff;">
                                <div style="color:#aa00ff;font-weight:bold;margin-bottom:6px;">🔬 UNIFIED SCANNER:</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Pairs Scanned:</span><span style="color:#fff;">${scanPairsCount}</span>
                                    <span style="color:#888;">Top Picks:</span><span style="color:#00ff88;">${scanTop}</span>
                                    <span style="color:#888;">Timeframe:</span><span style="color:#fff;">15m</span>
                                    <span style="color:#888;">Pre-filter:</span><span style="color:#00d4ff;">Active</span>
                                </div>
                                <div style="margin-top:6px;font-size:10px;color:#888;">
                                    Scans all USDT perpetual pairs through indicators, regime, volume, and session filters
                                </div>
                            </div>`;
                        break;
                    case 'regime':
                        const regimeColor = regime.regime && regime.regime.toUpperCase().includes('TRENDING') ? '#00ff88' : regime.regime && regime.regime.toUpperCase() === 'RANGING' ? '#ffcc00' : regime.regime && regime.regime.toUpperCase() === 'CHOPPY' ? '#ff6600' : '#00d4ff';
                        const trendStr = (regime.trend_strength || 0).toFixed(1);
                        const conf = ((regime.confidence || 0)).toFixed(0);
                        const volRatio = (regime.volatility_ratio || 0).toFixed(2);
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(255,204,0,0.1);border-radius:6px;border-left:3px solid #ffcc00;">
                                <div style="color:#ffcc00;font-weight:bold;margin-bottom:6px;">📊 MARKET REGIME:</div>
                                <div style="text-align:center;padding:8px;background:rgba(0,0,0,0.3);border-radius:6px;margin-bottom:8px;">
                                    <span style="color:${regimeColor};font-size:16px;font-weight:bold;text-transform:uppercase;">${regime.regime || 'UNKNOWN'}</span>
                                </div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Volatility:</span><span style="color:#fff;">${regime.volatility || 'unknown'}</span>
                                    <span style="color:#888;">Vol Ratio:</span><span style="color:#fff;">${volRatio}x</span>
                                    <span style="color:#888;">Trend Strength:</span><span style="color:#00ff88;">${trendStr}%</span>
                                    <span style="color:#888;">Confidence:</span><span style="color:#00d4ff;">${conf}%</span>
                                    <span style="color:#888;">ADX:</span><span style="color:#fff;">${(regime.adx || 0).toFixed(1)}</span>
                                    <span style="color:#888;">Hurst Exp:</span><span style="color:#fff;">${(regime.hurst || 0.5).toFixed(3)}</span>
                                </div>
                                <div style="margin-top:8px;padding:8px;background:rgba(0,0,0,0.3);border-radius:4px;font-size:10px;color:#aaa;">
                                    <div style="color:#ffcc00;font-weight:bold;margin-bottom:4px;">Status:</div>
                                    <div>${regime.tradeable ? '✅ Tradeable' : '⚠️ Not Tradeable'}</div>
                                    <div style="margin-top:4px;color:#888;font-size:9px;">${regime.description || 'Market regime analysis'}</div>
                                </div>
                            </div>`;
                        break;
                    case 'math':
                        const mathScoreVal = (pipelineComps.ai_filter && pipelineComps.ai_filter.last_score) || (ai && ai.last_score) || '--';
                        const mathConf = (ai && ai.last_confidence) || '--';
                        const penalties = data.penalties || {};
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(0,255,136,0.1);border-radius:6px;border-left:3px solid #00ff88;">
                                <div style="color:#00ff88;font-weight:bold;margin-bottom:6px;">🧮 PhD MATH SCORING:</div>
                                <div style="text-align:center;padding:8px;background:rgba(0,0,0,0.3);border-radius:6px;margin-bottom:8px;">
                                    <span style="color:#00ff88;font-size:18px;font-weight:bold;">${mathScoreVal !== '--' ? Math.round(mathScoreVal) + '/100' : '--'}</span>
                                </div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Components:</span><span style="color:#fff;">13 weighted</span>
                                    <span style="color:#888;">Confidence:</span><span style="color:#00d4ff;">${mathConf !== '--' ? (mathConf * 100).toFixed(0) + '%' : '--'}</span>
                                    <span style="color:#888;">Hurst Exp:</span><span style="color:#fff;">${(regime.hurst || 0.5).toFixed(3)}</span>
                                    <span style="color:#888;">Fractal Dim:</span><span style="color:#fff;">${indicators.fractal_dim ? indicators.fractal_dim.toFixed(3) : '--'}</span>
                                    <span style="color:#888;">Kelly Frac:</span><span style="color:#fff;">${indicators.kelly ? (indicators.kelly * 100).toFixed(1) + '%' : '--'}</span>
                                    <span style="color:#888;">Sharpe Ratio:</span><span style="color:#fff;">${indicators.sharpe ? indicators.sharpe.toFixed(2) : '--'}</span>
                                </div>
                                <div style="margin-top:6px;font-size:10px;color:#888;">
                                    RSI·ADX·ATR·MACD·BB·EMA·Hurst·Fractal·Kelly·Sharpe·Vol·MTF·Regime
                                </div>
                            </div>`;
                        break;
                    case 'gemini':
                        // Gemini AI + AI tracker historical data
                        const aiTracker = data.ai_tracker || {};
                        const aiTrackerSummary = aiTracker.summary || {};
                        const recentDecisions = aiTracker.recent_decisions || [];
                        const lastDecision = recentDecisions.length > 0 ? recentDecisions[0] : {};
                        
                        const trackerApprovalRate = aiTrackerSummary.approval_rate ? (aiTrackerSummary.approval_rate * 100).toFixed(0) : '--';
                        const trackerWinRate = aiTrackerSummary.approval_accuracy ? (aiTrackerSummary.approval_accuracy * 100).toFixed(1) : '--';
                        const netValue = aiTrackerSummary.net_ai_value || '--';
                        
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(66,133,244,0.1);border-radius:6px;border-left:3px solid #4285f4;">
                                <div style="color:#4285f4;font-weight:bold;margin-bottom:6px;">🤖 GEMINI 2.0 FLASH AI:</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Total Decisions:</span><span style="color:#fff;">${aiTrackerSummary.total_decisions || 0}</span>
                                    <span style="color:#888;">Approved:</span><span style="color:#00ff88;">${aiTrackerSummary.approved_count || 0}</span>
                                    <span style="color:#888;">Wins:</span><span style="color:#00ff88;">${aiTrackerSummary.approved_wins || 0}</span>
                                    <span style="color:#888;">Losses:</span><span style="color:#ff4444;">${aiTrackerSummary.approved_losses || 0}</span>
                                    <span style="color:#888;">Approval Rate:</span><span style="color:#00d4ff;">${trackerApprovalRate}%</span>
                                    <span style="color:#888;">Win Rate:</span><span style="color:#00ff88;">${trackerWinRate}%</span>
                                    <span style="color:#888;">Net P&L:</span><span style="color:${netValue.toString().startsWith('-') ? '#ff4444' : '#00ff88'};">${netValue}</span>
                                </div>
                                <div style="margin-top:8px;padding:8px;background:rgba(0,0,0,0.3);border-radius:4px;font-size:10px;color:#aaa;">
                                    <div style="color:#4285f4;font-weight:bold;margin-bottom:4px;">Last Decision:</div>
                                    <div>Direction: <span style="color:#fff;">${lastDecision.signal_direction || '--'}</span></div>
                                    <div>Symbol: <span style="color:#fff;">${lastDecision.symbol || '--'}</span></div>
                                    <div>Outcome: <span style="color:${lastDecision.trade_outcome === 'WIN' ? '#00ff88' : '#ff4444'};">${lastDecision.trade_outcome || '--'}</span></div>
                                </div>
                            </div>`;
                        break;
                    case 'execute':
                        const execHasPos = position && position.side;
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(255,255,0,0.1);border-radius:6px;border-left:3px solid #ffff00;">
                                <div style="color:#ffff00;font-weight:bold;margin-bottom:6px;">⚡ ORDER EXECUTION:</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Exchange:</span><span style="color:#fff;">Bybit</span>
                                    <span style="color:#888;">Type:</span><span style="color:#fff;">USDT Perpetual</span>
                                    <span style="color:#888;">Order Type:</span><span style="color:#fff;">Market + Limit SL/TP</span>
                                    <span style="color:#888;">Status:</span><span style="color:${execHasPos ? '#00ff88' : '#888'};">${execHasPos ? 'ACTIVE' : 'IDLE'}</span>
                                </div>
                                <div style="margin-top:6px;font-size:10px;color:#888;">
                                    Places market orders on Bybit with server-side stop-loss and take-profit
                                </div>
                            </div>`;
                        break;
                    case 'position':
                        const posHasPos = position && position.side;
                        const posColor = posHasPos ? (position.side === 'LONG' ? '#00ff88' : '#ff4444') : '#888';
                        const pnlColor = (position.pnl_percent || 0) >= 0 ? '#00ff88' : '#ff4444';
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(0,255,136,0.1);border-radius:6px;border-left:3px solid #00ff88;">
                                <div style="color:#00ff88;font-weight:bold;margin-bottom:6px;">💼 POSITION MONITOR:</div>
                                <div style="text-align:center;padding:8px;background:rgba(0,0,0,0.3);border-radius:6px;margin-bottom:8px;">
                                    <span style="color:${posColor};font-size:14px;font-weight:bold;">${posHasPos ? position.side + ' ' + (position.symbol || '') : 'NO POSITION'}</span>
                                </div>
                                ${posHasPos ? `
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Entry:</span><span style="color:#fff;">$${(position.entry || 0).toFixed(2)}</span>
                                    <span style="color:#888;">Current:</span><span style="color:#fff;">$${(position.current_price || 0).toFixed(2)}</span>
                                    <span style="color:#888;">Size:</span><span style="color:#fff;">$${(position.size || 0).toFixed(2)}</span>
                                    <span style="color:#888;">PnL:</span><span style="color:${pnlColor};">${(position.pnl_percent || 0) >= 0 ? '+' : ''}${(position.pnl_percent || 0).toFixed(2)}%</span>
                                    <span style="color:#888;">Stop Loss:</span><span style="color:#ff4444;">$${(position.stop_loss || 0).toFixed(2)}</span>
                                    <span style="color:#888;">Take Profit:</span><span style="color:#00ff88;">$${(position.take_profit || 0).toFixed(2)}</span>
                                </div>` : '<div style="color:#666;font-size:11px;text-align:center;">Monitoring trailing stop, TP hits, exit signals</div>'}
                            </div>`;
                        break;
                    case 'telegram':
                        const teleStatus = data.status || {};
                        liveSection = `
                            <div style="margin-top:10px;padding:10px;background:rgba(0,136,204,0.1);border-radius:6px;border-left:3px solid #0088cc;">
                                <div style="color:#0088cc;font-weight:bold;margin-bottom:6px;">📱 TELEGRAM STATUS:</div>
                                <div style="text-align:center;padding:8px;background:rgba(0,0,0,0.3);border-radius:6px;margin-bottom:8px;">
                                    <span style="color:#00ff88;font-size:14px;font-weight:bold;">CONNECTED</span>
                                </div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;">
                                    <span style="color:#888;">Bot Active:</span><span style="color:#00ff88;">YES</span>
                                    <span style="color:#888;">Notifications:</span><span style="color:#00ff88;">ENABLED</span>
                                    <span style="color:#888;">Commands:</span><span style="color:#00d4ff;">/start, /status, /balance</span>
                                </div>
                                <div style="margin-top:6px;font-size:10px;color:#888;">
                                    Sends trade alerts, position updates, and daily summaries
                                </div>
                            </div>`;
                        break;
                    default:
                        liveSection = '';
                }
                
                return `
                    <div style="font-size:16px;margin-bottom:8px;">${node.icon} <span style="color:${node.color};font-weight:bold;text-shadow:0 0 10px ${node.color};">${node.desc}</span></div>
                    <div style="margin:8px 0;padding:8px;background:rgba(0,100,0,0.15);border-radius:6px;border-left:3px solid #00ff88;">
                        <div style="color:#00ff88;font-weight:bold;margin-bottom:4px;font-size:11px;">📥 INPUTS:</div>
                        <div style="color:#aaa;font-size:11px;">${node.inputs.join(' • ')}</div>
                    </div>
                    <div style="padding:8px;background:rgba(100,100,0,0.15);border-radius:6px;border-left:3px solid #ffcc00;">
                        <div style="color:#ffcc00;font-weight:bold;margin-bottom:4px;font-size:11px;">📤 OUTPUTS:</div>
                        <div style="color:#aaa;font-size:11px;">${node.outputs.join(' • ')}</div>
                    </div>
                    ${liveSection}
                `;
            }
            
            // Click handler for nodes
            canvas.addEventListener('click', function(e) {
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                // Use global machineNodes if available
                const nodesForClick = window.machineNodes || nodes;
                
                // Check if click is on any node
                for (const [key, node] of Object.entries(nodesForClick)) {
                    const size = (node.size || 1) * 25;
                    const dist = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
                    if (dist < size + 10) {
                        // Toggle: if same node clicked, hide
                        if (currentNodeKey === key && tooltip.style.display === 'block') {
                            tooltip.style.display = 'none';
                            currentNodeKey = null;
                            return;
                        }
                        
                        // Show tooltip with live data
                        currentNodeKey = key;
                        tooltip.innerHTML = generateNodeTooltip(key, node);
                        tooltip.style.display = 'block';
                        tooltip.style.maxWidth = '320px';
                        // Position using fixed coordinates (relative to viewport)
                        const nodeScreenX = rect.left + node.x;
                        const nodeScreenY = rect.top + node.y;
                        // Position tooltip above node if bottom half, below if top half
                        const tooltipHeight = 200;
                        let tooltipY = nodeScreenY - tooltipHeight - 10;
                        if (tooltipY < 10) tooltipY = nodeScreenY + 40;
                        tooltip.style.left = Math.min(nodeScreenX + 30, window.innerWidth - 340) + 'px';
                        tooltip.style.top = tooltipY + 'px';
                        tooltip.style.borderColor = node.color;
                        tooltip.style.boxShadow = '0 0 25px ' + node.color + '60';
                        return;
                    }
                }
                // Click outside nodes - hide tooltip
                tooltip.style.display = 'none';
                currentNodeKey = null;
            });
            
            // Hide tooltip on mouse leave
            canvas.addEventListener('mouseleave', function() {
                tooltip.style.display = 'none';
                currentNodeKey = null;
            });
            
            // Change cursor on hover over nodes
            canvas.addEventListener('mousemove', function(e) {
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                let onNode = false;
                for (const node of Object.values(nodes)) {
                    const size = (node.size || 1) * 25;
                    const dist = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
                    if (dist < size + 10) {
                        onNode = true;
                        break;
                    }
                }
                canvas.style.cursor = onNode ? 'pointer' : 'default';
            });
            
            // Connection paths - Real Julaba Pipeline Flow
            // Bybit → OHLCV → [IND, REG] → SCAN → PhD → AI → [RISK, EXEC] → POS → TELE
            const connections = [
                // Data Ingestion
                { from: 'bybit', to: 'ohlcv', speed: 2.5, layer: 0 },
                
                // OHLCV splits to Indicators and Regime
                { from: 'ohlcv', to: 'indicators', speed: 2, layer: 1 },
                { from: 'ohlcv', to: 'regime', speed: 2, layer: 1 },
                
                // Both feed into Scanner
                { from: 'indicators', to: 'scanner', speed: 1.5, layer: 2 },
                { from: 'regime', to: 'scanner', speed: 1.5, layer: 2 },
                
                // Scanner to PhD Math Scoring
                { from: 'scanner', to: 'math', speed: 1.2, layer: 3 },
                
                // PhD Math to Gemini AI
                { from: 'math', to: 'gemini', speed: 1, layer: 4 },
                
                // AI splits to Risk and Execution
                { from: 'gemini', to: 'risk', speed: 1.5, layer: 5 },
                { from: 'gemini', to: 'execute', speed: 1.5, layer: 5 },
                
                // Risk also feeds Execution
                { from: 'risk', to: 'execute', speed: 2, layer: 5 },
                
                // Both merge to Position Management
                { from: 'risk', to: 'position', speed: 1.5, layer: 6 },
                { from: 'execute', to: 'position', speed: 1.5, layer: 6 },
                
                // Position to Telegram
                { from: 'position', to: 'telegram', speed: 2, layer: 7 }
            ];
            
            // Create flowing particles - PIPELINE STYLE
            // Data flows in waves: Input → Technical → Analysis → AI → Decision → Output
            function createParticle(conn, waveOffset) {
                const from = nodes[conn.from];
                const to = nodes[conn.to];
                return {
                    x: from.x,
                    y: from.y,
                    targetX: to.x,
                    targetY: to.y,
                    progress: 0,
                    speed: conn.speed * 0.008,  // Faster for pipeline feel
                    color: from.color,
                    size: 4,  // Larger data packets
                    layer: conn.layer,
                    waveId: waveOffset || 0,
                    isDataPacket: true
                };
            }
            
            // Pipeline wave system - data flows through in coordinated waves
            let pipelineWaveTime = 0;
            let pipelineWaves = [];
            const WAVE_INTERVAL = 3; // New wave every 3 seconds
            
            // Make createPipelineWave globally accessible for data updates
            window.createPipelineWave = function() {
                const waveId = Date.now();
                const wave = {
                    id: waveId,
                    particles: [],
                    stage: 0,  // 0=bybit→ohlcv, 1=ohlcv→ind/reg, 2=ind/reg→scan, 3=scan→math, 4=math→ai, 5=ai→risk/exec, 6=risk/exec→pos, 7=pos→tele
                    startTime: Date.now()
                };
                
                // Create particles for layer 0 (input to technical)
                connections.filter(c => c.layer === 0).forEach(conn => {
                    const p = createParticle(conn, waveId);
                    p.waveStage = 0;
                    wave.particles.push({ ...p, conn });
                });
                
                pipelineWaves.push(wave);
                
                // Limit waves in memory
                if (pipelineWaves.length > 5) {
                    pipelineWaves.shift();
                }
            };
            
            // Local reference
            const createPipelineWave = window.createPipelineWave;
            
            // Initialize first wave
            createPipelineWave();
            
            // Initialize some static flow particles for continuous effect
            connections.forEach((conn, i) => {
                for (let j = 0; j < 2; j++) {
                    const p = createParticle(conn);
                    p.progress = (j / 2) + (i * 0.1) % 1;
                    p.isBackgroundFlow = true;
                    p.size = 2;  // Smaller background particles
                    dataParticles.push({ ...p, conn });
                }
            });
            
            // Draw circuit board background pattern
            function drawCircuitBackground() {
                ctx.strokeStyle = 'rgba(0, 212, 255, 0.03)';
                ctx.lineWidth = 1;
                
                // Horizontal circuit lines
                for (let y = 30; y < H; y += 40) {
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    for (let x = 0; x < W; x += 20) {
                        if (Math.random() > 0.7) {
                            ctx.lineTo(x, y + (Math.random() > 0.5 ? 5 : -5));
                        }
                        ctx.lineTo(x + 20, y);
                    }
                    ctx.stroke();
                }
                
                // Floating particles
                bgParticles.forEach(p => {
                    p.x += p.vx;
                    p.y += p.vy;
                    if (p.x < 0 || p.x > W) p.vx *= -1;
                    if (p.y < 0 || p.y > H) p.vy *= -1;
                    
                    ctx.fillStyle = p.color;
                    ctx.globalAlpha = p.alpha * (0.5 + Math.sin(time * 2 + p.x * 0.01) * 0.5);
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                    ctx.fill();
                });
                ctx.globalAlpha = 1;
            }
            
            // Neural synapse fire effect
            function fireNeuralSynapse(fromNode, toNode, color) {
                neuralSynapses.push({
                    from: fromNode,
                    to: toNode,
                    progress: 0,
                    color: color || '#fff',
                    width: 4,
                    life: 1
                });
            }
            
            // Draw and update neural synapses
            function drawNeuralSynapses() {
                neuralSynapses = neuralSynapses.filter(s => {
                    s.progress += 0.03;
                    s.life -= 0.02;
                    if (s.life <= 0) return false;
                    
                    const from = nodes[s.from];
                    const to = nodes[s.to];
                    if (!from || !to) return false;
                    
                    const midX = (from.x + to.x) / 2;
                    const midY = (from.y + to.y) / 2 + (from.y - to.y) * 0.2;
                    
                    // Calculate point along curve
                    const t = Math.min(s.progress, 1);
                    const x = (1-t)*(1-t)*from.x + 2*(1-t)*t*midX + t*t*to.x;
                    const y = (1-t)*(1-t)*from.y + 2*(1-t)*t*midY + t*t*to.y;
                    
                    // Draw electric arc
                    ctx.strokeStyle = s.color;
                    ctx.lineWidth = s.width * s.life;
                    ctx.shadowColor = s.color;
                    ctx.shadowBlur = 20;
                    ctx.globalAlpha = s.life;
                    
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    
                    // Add jagged lightning effect
                    const segments = 8;
                    for (let i = 1; i <= segments; i++) {
                        const segT = (i / segments) * t;
                        const segX = (1-segT)*(1-segT)*from.x + 2*(1-segT)*segT*midX + segT*segT*to.x;
                        const segY = (1-segT)*(1-segT)*from.y + 2*(1-segT)*segT*midY + segT*segT*to.y;
                        const jitter = (Math.random() - 0.5) * 10 * s.life;
                        ctx.lineTo(segX + jitter, segY + jitter);
                    }
                    ctx.stroke();
                    
                    // Bright head
                    ctx.fillStyle = '#fff';
                    ctx.beginPath();
                    ctx.arc(x, y, 6 * s.life, 0, Math.PI * 2);
                    ctx.fill();
                    
                    ctx.globalAlpha = 1;
                    ctx.shadowBlur = 0;
                    return true;
                });
            }
            
            // Trigger random neural fires
            function triggerRandomSynapses() {
                if (Math.random() < 0.02) {
                    const connIdx = Math.floor(Math.random() * connections.length);
                    const conn = connections[connIdx];
                    fireNeuralSynapse(conn.from, conn.to, nodes[conn.from].color);
                }
            }
            
            function drawNode(node, x, y, nodeKey) {
                const size = (node.size || 1) * 25;
                const pulseSize = size + Math.sin(time * 3) * 4;
                
                // Check for highlight state from events
                var highlightState = window.machineNodeStates && window.machineNodeStates[nodeKey];
                var isHighlighted = highlightState && Date.now() < highlightState.until;
                var highlightColor = isHighlighted ? highlightState.color : null;
                var nodeColor = highlightColor || node.color;
                
                // Check for error/warning states
                var hasError = window.machineNodeErrors && window.machineNodeErrors[nodeKey];
                var hasWarning = window.machineNodeWarnings && window.machineNodeWarnings[nodeKey];
                
                // Error state - pulsing red ring
                if (hasError) {
                    const errorPulse = Math.sin(time * 8) * 0.4 + 0.6;
                    ctx.strokeStyle = `rgba(255, 68, 68, ${errorPulse})`;
                    ctx.lineWidth = 3;
                    ctx.shadowColor = '#ff4444';
                    ctx.shadowBlur = 15 + errorPulse * 10;
                    ctx.setLineDash([5, 5]);
                    ctx.lineDashOffset = time * 20;
                    ctx.beginPath();
                    ctx.arc(x, y, size + 8, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.shadowBlur = 0;
                    nodeColor = '#ff4444';  // Override node color
                }
                
                // Warning state - pulsing yellow ring
                if (hasWarning && !hasError) {
                    const warnPulse = Math.sin(time * 5) * 0.3 + 0.7;
                    ctx.strokeStyle = `rgba(255, 204, 0, ${warnPulse})`;
                    ctx.lineWidth = 2;
                    ctx.shadowColor = '#ffcc00';
                    ctx.shadowBlur = 10;
                    ctx.beginPath();
                    ctx.arc(x, y, size + 6, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                    nodeColor = '#ffcc00';  // Override node color
                }
                
                // Gemini AI + PhD Math special heartbeat effect
                if ((nodeKey === 'gemini' || nodeKey === 'math') && !hasError) {
                    const heartbeat = Math.sin(time * 6) * 0.5 + 0.5;
                    const hbColor = nodeKey === 'gemini' ? '66, 133, 244' : '0, 255, 136';
                    const heartGlow = ctx.createRadialGradient(x, y, 0, x, y, pulseSize * 3);
                    heartGlow.addColorStop(0, `rgba(${hbColor}, ${0.3 * heartbeat})`);
                    heartGlow.addColorStop(0.5, `rgba(${hbColor}, ${0.1 * heartbeat})`);
                    heartGlow.addColorStop(1, 'transparent');
                    ctx.fillStyle = heartGlow;
                    ctx.beginPath();
                    ctx.arc(x, y, pulseSize * 3, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Rotating hexagonal outline
                    ctx.strokeStyle = nodeKey === 'gemini' ? 'rgba(66, 133, 244, 0.3)' : 'rgba(0, 255, 136, 0.3)';
                    ctx.lineWidth = 1;
                    ctx.save();
                    ctx.translate(x, y);
                    ctx.rotate(time * 0.5);
                    ctx.beginPath();
                    for (let i = 0; i < 6; i++) {
                        const angle = (i / 6) * Math.PI * 2;
                        const hx = Math.cos(angle) * (size * 1.8);
                        const hy = Math.sin(angle) * (size * 1.8);
                        i === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
                    }
                    ctx.closePath();
                    ctx.stroke();
                    ctx.restore();
                }
                
                // Extra large glow when highlighted
                if (isHighlighted) {
                    const bigGlow = ctx.createRadialGradient(x, y, 0, x, y, pulseSize * 4);
                    bigGlow.addColorStop(0, highlightColor + 'aa');
                    bigGlow.addColorStop(0.3, highlightColor + '60');
                    bigGlow.addColorStop(0.6, highlightColor + '20');
                    bigGlow.addColorStop(1, 'transparent');
                    ctx.fillStyle = bigGlow;
                    ctx.beginPath();
                    ctx.arc(x, y, pulseSize * 4, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Fire synapses when highlighted
                    if (Math.random() < 0.1) {
                        connections.filter(c => c.from === nodeKey).forEach(c => {
                            fireNeuralSynapse(c.from, c.to, highlightColor);
                        });
                    }
                }
                
                // Outer glow (more vibrant)
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, pulseSize * 2);
                gradient.addColorStop(0, nodeColor + '60');
                gradient.addColorStop(0.4, nodeColor + '30');
                gradient.addColorStop(0.7, nodeColor + '10');
                gradient.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, pulseSize * 2, 0, Math.PI * 2);
                ctx.fill();
                
                // Main circle with glass effect
                ctx.fillStyle = isHighlighted ? 'rgba(30, 40, 60, 0.9)' : 'rgba(10, 10, 30, 0.7)';
                ctx.strokeStyle = nodeColor;
                ctx.lineWidth = isHighlighted ? 3 : 2;
                ctx.shadowColor = nodeColor;
                ctx.shadowBlur = isHighlighted ? 25 : 15;
                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                ctx.shadowBlur = 0;
                
                // Icon
                ctx.font = `${size * 0.7}px Arial`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(node.icon, x, y - 2);
                
                // Data flow indicator on right side
                const dataFlow = window.dataFlow || [];
                const incomingFlows = dataFlow.filter(f => f.to === nodeKey);
                const hasActiveIncoming = incomingFlows.some(f => f.active);
                
                if (incomingFlows.length > 0) {
                    const indicatorX = x + size + 8;
                    const indicatorY = y;
                    const indicatorSize = 4;
                    
                    if (hasActiveIncoming) {
                        // Green dot - data flowing in
                        ctx.fillStyle = '#00ff88';
                        ctx.shadowColor = '#00ff88';
                        ctx.shadowBlur = 8;
                        ctx.beginPath();
                        ctx.arc(indicatorX, indicatorY, indicatorSize, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.shadowBlur = 0;
                    } else {
                        // Red X - connections exist but no flow
                        ctx.strokeStyle = '#ff4444';
                        ctx.lineWidth = 1;
                        const offset = indicatorSize;
                        ctx.beginPath();
                        ctx.moveTo(indicatorX - offset, indicatorY - offset);
                        ctx.lineTo(indicatorX + offset, indicatorY + offset);
                        ctx.moveTo(indicatorX - offset, indicatorY + offset);
                        ctx.lineTo(indicatorX + offset, indicatorY - offset);
                        ctx.stroke();
                    }
                }
                
                // Label with glow
                ctx.font = 'bold 9px Orbitron, sans-serif';
                ctx.fillStyle = nodeColor;
                ctx.shadowColor = nodeColor;
                ctx.shadowBlur = isHighlighted ? 15 : 8;
                ctx.fillText(node.label, x, y + size + 12);
                ctx.shadowBlur = 0;
                
                // Sub-label (vibrant)
                if (node.sub) {
                    ctx.font = 'bold 7px Rajdhani, sans-serif';
                    ctx.fillStyle = '#aaa';
                    ctx.fillText(node.sub, x, y + size + 23);
                }
            }
            
            function drawConnection(from, to, color, layer, fromKey, toKey) {
                // Check if data is actually flowing through this connection
                const dataFlow = window.dataFlow || [];
                const flowConnection = dataFlow.find(f => f.from === fromKey && f.to === toKey);
                const isFlowing = flowConnection && flowConnection.active;
                
                // Check if position is active for enhanced flow
                const data = window.lastDashboardData || {};
                const positions = data.open_position || data.positions || [];
                const hasPosition = positions.length > 0 || (positions && positions.symbol);
                const positionSide = hasPosition ? (positions[0]?.side || positions.side || 'LONG') : null;
                const isLongPosition = positionSide === 'LONG';
                
                // Calculate curve points
                const midX = (from.x + to.x) / 2;
                const midY = (from.y + to.y) / 2 + (from.y - to.y) * 0.2;
                
                // If data not flowing, show dimmed but visible connection
                if (!isFlowing) {
                    // Base line - more visible
                    ctx.strokeStyle = 'rgba(60, 80, 120, 0.4)';
                    ctx.lineWidth = 2;
                    ctx.shadowColor = 'rgba(0, 150, 255, 0.2)';
                    ctx.shadowBlur = 5;
                    ctx.setLineDash([]);
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    
                    // Dashed overlay
                    ctx.strokeStyle = 'rgba(100, 120, 150, 0.3)';
                    ctx.lineWidth = 1;
                    ctx.shadowBlur = 0;
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    return;
                }
                
                // Pulsing connection intensity based on layer
                const pulse = Math.sin(time * 2 + layer * 0.5) * 0.3 + 0.7;
                
                // ═══════════════════════════════════════════════════════════════
                // 🌊 WATER FLOW EFFECT - Enhanced when position is active
                // ═══════════════════════════════════════════════════════════════
                if (hasPosition) {
                    // Draw glowing water stream background (wider, more visible)
                    const flowColor = isLongPosition ? 'rgba(0,255,136,' : 'rgba(255,68,68,';
                    const accentColor = isLongPosition ? '#00ff88' : '#ff4444';
                    
                    // Layer 0: Base solid line for visibility
                    ctx.strokeStyle = flowColor + '0.6)';
                    ctx.lineWidth = 3;
                    ctx.shadowColor = accentColor;
                    ctx.shadowBlur = 25;
                    ctx.lineCap = 'round';
                    ctx.setLineDash([]);
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    
                    // Layer 1: Wide outer glow (the "river bed")
                    ctx.strokeStyle = flowColor + (0.25 * pulse) + ')';
                    ctx.lineWidth = 14;
                    ctx.shadowBlur = 30;
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    
                    // Layer 2: Medium flow stream
                    ctx.strokeStyle = flowColor + (0.5 * pulse) + ')';
                    ctx.lineWidth = 8;
                    ctx.shadowBlur = 20;
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    
                    // Layer 3: Bright core stream with flowing dashes
                    ctx.strokeStyle = flowColor + (0.7 * pulse) + ')';
                    ctx.lineWidth = 3;
                    ctx.shadowBlur = 10;
                    ctx.setLineDash([8, 4]);
                    ctx.lineDashOffset = -time * 80;  // Fast flow
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    
                    // Layer 4: White highlight (water sparkle)
                    ctx.strokeStyle = 'rgba(255,255,255,' + (0.3 * pulse) + ')';
                    ctx.lineWidth = 1.5;
                    ctx.shadowColor = '#fff';
                    ctx.shadowBlur = 8;
                    ctx.setLineDash([3, 12]);
                    ctx.lineDashOffset = -time * 120;  // Faster sparkle
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    
                    ctx.setLineDash([]);
                    ctx.shadowBlur = 0;
                    ctx.lineCap = 'butt';
                } else {
                    // Standard flowing connection (scanning mode) - MORE VISIBLE
                    // Base solid line
                    ctx.strokeStyle = color + 'aa';
                    ctx.lineWidth = 2.5;
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 12;
                    ctx.lineCap = 'round';
                    ctx.setLineDash([]);
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    
                    // Flowing dashes on top
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1.5;
                    ctx.shadowBlur = 8 + pulse * 6;
                    ctx.setLineDash([6, 4]);
                    ctx.lineDashOffset = -time * 60;
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.quadraticCurveTo(midX, midY, to.x, to.y);
                    ctx.stroke();
                    
                    ctx.setLineDash([]);
                    ctx.shadowBlur = 0;
                    ctx.lineCap = 'butt';
                }
            }
            
            function drawParticle(p) {
                // Bezier curve position
                const from = nodes[p.conn.from];
                const to = nodes[p.conn.to];
                const midX = (from.x + to.x) / 2;
                const midY = (from.y + to.y) / 2 + (from.y - to.y) * 0.2;
                
                const t = Math.min(p.progress, 1);
                const x = (1-t)*(1-t)*from.x + 2*(1-t)*t*midX + t*t*to.x;
                const y = (1-t)*(1-t)*from.y + 2*(1-t)*t*midY + t*t*to.y;
                
                // Trail effect - longer trail for data packets
                const trailLength = p.isDataPacket ? 0.12 : 0.05;
                const prevT = Math.max(0, t - trailLength);
                const prevX = (1-prevT)*(1-prevT)*from.x + 2*(1-prevT)*prevT*midX + prevT*prevT*to.x;
                const prevY = (1-prevT)*(1-prevT)*from.y + 2*(1-prevT)*prevT*midY + prevT*prevT*to.y;
                
                // Draw trail
                const trailGradient = ctx.createLinearGradient(prevX, prevY, x, y);
                trailGradient.addColorStop(0, 'transparent');
                trailGradient.addColorStop(0.5, p.color + '40');
                trailGradient.addColorStop(1, p.color + 'aa');
                ctx.strokeStyle = trailGradient;
                ctx.lineWidth = p.isDataPacket ? 3 : 2;
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(prevX, prevY);
                ctx.lineTo(x, y);
                ctx.stroke();
                
                // Data packet appearance (larger glowing orb with pulsing)
                const particleSize = p.size || 3;
                const pulseScale = p.isDataPacket ? 1 + Math.sin(time * 8 + p.progress * 10) * 0.2 : 1;
                const displaySize = particleSize * pulseScale;
                
                // Outer glow
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, displaySize * 5);
                gradient.addColorStop(0, '#fff');
                gradient.addColorStop(0.15, p.color);
                gradient.addColorStop(0.4, p.color + '80');
                gradient.addColorStop(0.7, p.color + '30');
                gradient.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, displaySize * 5, 0, Math.PI * 2);
                ctx.fill();
                
                // Bright core with shadow
                ctx.shadowColor = p.color;
                ctx.shadowBlur = p.isDataPacket ? 15 : 10;
                ctx.fillStyle = '#fff';
                ctx.beginPath();
                ctx.arc(x, y, displaySize * (p.isDataPacket ? 2 : 1.5), 0, Math.PI * 2);
                ctx.fill();
                
                // Inner glow ring for data packets
                if (p.isDataPacket) {
                    ctx.strokeStyle = p.color;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.arc(x, y, displaySize * 3, 0, Math.PI * 2);
                    ctx.stroke();
                }
                
                ctx.shadowBlur = 0;
            }
            
            // === ACTIVE TRADE EFFECTS ===
            let tradeRings = [];
            let tradeSparks = [];
            let energyBolts = [];
            
            function drawActiveTradeEffects() {
                if (!position) return;
                
                // Check if data is actually flowing through the pipeline
                const dataFlow = window.dataFlow || [];
                const aiToRiskFlow = dataFlow.find(f => f.from === 'ai_filter' && f.to === 'risk_manager');
                const riskToTradeFlow = dataFlow.find(f => f.from === 'risk_manager' && f.to === 'position_monitor');
                
                // Only show animation if data is actively flowing
                const dataFlowing = aiToRiskFlow && aiToRiskFlow.active && riskToTradeFlow && riskToTradeFlow.active;
                
                if (!dataFlowing) {
                    // Show dimmed state if no data flow
                    const execNode = nodes.execute;
                    ctx.fillStyle = 'rgba(255, 0, 100, 0.1)';
                    ctx.beginPath();
                    ctx.arc(execNode.x, execNode.y, 50, 0, Math.PI * 2);
                    ctx.fill();
                    return;
                }
                
                const tradeNode = nodes.execute;
                const x = tradeNode.x;
                const y = tradeNode.y;
                
                // 1. EXPANDING RINGS (pulse outward)
                if (Math.random() < 0.03) {
                    tradeRings.push({ x, y, radius: 30, alpha: 1, color: '#00ff88' });
                }
                tradeRings = tradeRings.filter(ring => {
                    ring.radius += 2;
                    ring.alpha -= 0.02;
                    if (ring.alpha <= 0) return false;
                    
                    ctx.strokeStyle = `rgba(0, 255, 136, ${ring.alpha})`;
                    ctx.lineWidth = 2;
                    ctx.shadowColor = '#00ff88';
                    ctx.shadowBlur = 10;
                    ctx.beginPath();
                    ctx.arc(ring.x, ring.y, ring.radius, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                    return true;
                });
                
                // 2. SPARKLING PARTICLES around trade node
                if (Math.random() < 0.15) {
                    const angle = Math.random() * Math.PI * 2;
                    const dist = 40 + Math.random() * 30;
                    tradeSparks.push({
                        x: x + Math.cos(angle) * dist,
                        y: y + Math.sin(angle) * dist,
                        vx: (Math.random() - 0.5) * 2,
                        vy: (Math.random() - 0.5) * 2,
                        life: 1,
                        size: 2 + Math.random() * 3,
                        color: Math.random() > 0.5 ? '#00ff88' : '#ffff00'
                    });
                }
                tradeSparks = tradeSparks.filter(spark => {
                    spark.x += spark.vx;
                    spark.y += spark.vy;
                    spark.life -= 0.02;
                    if (spark.life <= 0) return false;
                    
                    ctx.fillStyle = spark.color;
                    ctx.shadowColor = spark.color;
                    ctx.shadowBlur = 8;
                    ctx.globalAlpha = spark.life;
                    ctx.beginPath();
                    ctx.arc(spark.x, spark.y, spark.size, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.globalAlpha = 1;
                    ctx.shadowBlur = 0;
                    return true;
                });
                
                // 3. ENERGY BOLTS from Gemini AI to Execute (only if data flowing)
                if (Math.random() < 0.02) {
                    const startNode = nodes.gemini;
                    energyBolts.push({
                        points: generateLightningPath(startNode.x, startNode.y, x, y),
                        life: 1,
                        color: '#00ff88'
                    });
                }
                energyBolts = energyBolts.filter(bolt => {
                    bolt.life -= 0.05;
                    if (bolt.life <= 0) return false;
                    
                    ctx.strokeStyle = `rgba(0, 255, 136, ${bolt.life})`;
                    ctx.lineWidth = 2;
                    ctx.shadowColor = '#00ff88';
                    ctx.shadowBlur = 15;
                    ctx.beginPath();
                    ctx.moveTo(bolt.points[0].x, bolt.points[0].y);
                    bolt.points.forEach(pt => ctx.lineTo(pt.x, pt.y));
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                    return true;
                });
                
                // 4. PULSING GLOW on trade node (heartbeat)
                const pulse = Math.sin(time * 5) * 0.3 + 0.7;
                const glowSize = 60 + pulse * 20;
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, glowSize);
                gradient.addColorStop(0, `rgba(0, 255, 136, ${0.4 * pulse})`);
                gradient.addColorStop(0.5, `rgba(0, 255, 136, ${0.15 * pulse})`);
                gradient.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, glowSize, 0, Math.PI * 2);
                ctx.fill();
                
                // 5. ROTATING ORBIT RING
                ctx.strokeStyle = 'rgba(0, 255, 136, 0.5)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 6]);
                ctx.lineDashOffset = time * 30;
                ctx.beginPath();
                ctx.arc(x, y, 50, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
                
                // 6. ORBITING DOT
                const orbitAngle = time * 2;
                const orbitX = x + Math.cos(orbitAngle) * 50;
                const orbitY = y + Math.sin(orbitAngle) * 50;
                ctx.fillStyle = '#00ff88';
                ctx.shadowColor = '#00ff88';
                ctx.shadowBlur = 10;
                ctx.beginPath();
                ctx.arc(orbitX, orbitY, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.shadowBlur = 0;
            }
            
            function generateLightningPath(x1, y1, x2, y2) {
                const points = [{x: x1, y: y1}];
                const segments = 5;
                for (let i = 1; i < segments; i++) {
                    const t = i / segments;
                    const x = x1 + (x2 - x1) * t + (Math.random() - 0.5) * 30;
                    const y = y1 + (y2 - y1) * t + (Math.random() - 0.5) * 20;
                    points.push({x, y});
                }
                points.push({x: x2, y: y2});
                return points;
            }
            
            function animate() {
              try {
                // Clear with transparency (let background show through)
                ctx.clearRect(0, 0, W, H);
                
                time += 0.016;
                
                // Use global machineNodes if available (gets updated with live data)
                const nodesForRender = window.machineNodes || nodes;
                
                // === UPDATE NODE SUB-LABELS WITH LIVE DATA EACH FRAME ===
                // This ensures values are always current
                if (window.lastPipelineData || window.lastData) {
                    var pipelineComps = (window.lastPipelineData && window.lastPipelineData.components) || {};
                    var dataFlow = (window.lastPipelineData && window.lastPipelineData.data_flow) || [];
                    var mainData = window.lastData || {};
                    var status = mainData.status || {};
                    var indicators = mainData.indicators || {};
                    var regime = mainData.regime || {};
                    var ai = mainData.ai || {};
                    var ml = mainData.ml || mainData.ml_stats || {};
                    var position = mainData.open_position || {};
                    var risk = mainData.risk || {};
                    var isConnected = status.connected !== false;
                    
                    // Expose data flow globally for animation control
                    window.dataFlow = dataFlow;
                    
                    // BYBIT - show price
                    var marketPrice = (pipelineComps.market_feed && pipelineComps.market_feed.last_price) || status.current_price;
                    nodesForRender.bybit.sub = marketPrice ? '$' + parseFloat(marketPrice).toFixed(1) : (isConnected ? 'OK' : 'ERR');
                    
                    // OHLCV - show close price from last candle
                    var ohlcList = mainData.ohlc && Array.isArray(mainData.ohlc) ? mainData.ohlc : [];
                    var lastCandle = ohlcList.length > 0 ? ohlcList[ohlcList.length - 1] : null;
                    var candleClose = lastCandle ? parseFloat(lastCandle.c).toFixed(3) : '--';
                    nodesForRender.ohlcv.sub = candleClose !== '--' ? 'C:' + candleClose : '15m';
                    
                    // Indicators - show RSI value
                    var indRsi = (pipelineComps.indicators && pipelineComps.indicators.rsi) || indicators.rsi;
                    nodesForRender.indicators.sub = indRsi ? 'RSI ' + Math.round(indRsi) : '--';
                    
                    // Regime - show current regime with confidence
                    var regimeText = (pipelineComps.regime && pipelineComps.regime.current) || regime.regime || 'UNKNOWN';
                    var regimeConf = regime && regime.confidence ? Math.round(regime.confidence) : 0;
                    nodesForRender.regime.sub = regimeConf > 0 ? regimeConf + '%' : regimeText.substring(0, 4).toUpperCase();
                    
                    // Scanner - show pairs scanned
                    var scanCount = (pipelineComps.scanner && pipelineComps.scanner.pairs_scanned) || 50;
                    nodesForRender.scanner.sub = scanCount + ' PAIRS';
                    
                    // PhD Math - show score
                    var mathScore = (pipelineComps.ai_filter && pipelineComps.ai_filter.last_score) || (ai && ai.last_score);
                    nodesForRender.math.sub = mathScore ? Math.round(mathScore) + '/100' : 'RDY';
                    
                    // Gemini AI - show mode
                    var aiMode = (pipelineComps.ai_filter && pipelineComps.ai_filter.mode) || (ai && ai.mode);
                    var aiCalls = (pipelineComps.ai_filter && pipelineComps.ai_filter.call_count);
                    nodesForRender.gemini.sub = aiMode ? aiMode.substring(0, 4).toUpperCase() : (aiCalls ? '#' + aiCalls : 'RDY');
                    
                    // Risk - show can_trade
                    var riskCanTrade = (pipelineComps.risk_manager && pipelineComps.risk_manager.can_trade) !== false;
                    nodesForRender.risk.sub = riskCanTrade ? 'OK' : 'STOP';
                    
                    // Execute - show status
                    var posMonitor = pipelineComps.position_monitor;
                    var posStatus = (posMonitor && posMonitor.has_position) ? 'ACTIVE' : 'IDLE';
                    nodesForRender.execute.sub = posStatus;
                    
                    // Position - show side
                    var posSide = (posMonitor && posMonitor.has_position) ? posMonitor.position_side.toUpperCase() : 
                                  (position && position.side ? position.side.toUpperCase() : 'NIL');
                    nodesForRender.position.sub = posSide;
                    
                    // Telegram
                    var teleComp = pipelineComps.telegram;
                    nodesForRender.telegram.sub = (teleComp && (teleComp.connected || teleComp.is_started)) ? 'ON' : 'OFF';
                }
                
                // Draw circuit background
                drawCircuitBackground();
                
                // Draw active trade effects FIRST (behind nodes)
                drawActiveTradeEffects();
                
                // Trigger random neural synapses
                triggerRandomSynapses();
                
                // Draw connections first (behind everything)
                connections.forEach(conn => {
                    drawConnection(nodesForRender[conn.from], nodesForRender[conn.to], nodesForRender[conn.from].color, conn.layer, conn.from, conn.to);
                });
                
                // Draw neural synapses (electric arcs)
                drawNeuralSynapses();
                
                // Draw nodes (pass key for highlight lookup)
                Object.entries(nodesForRender).forEach(([key, node]) => {
                    drawNode(node, node.x, node.y, key);
                });
                
                // Draw and update particles - PIPELINE WAVE SYSTEM
                // Only animate if data is actually flowing
                const isDataFlowing = window.machineDataFlowing !== false;
                
                // Create new pipeline wave periodically (only if data flowing)
                pipelineWaveTime += 0.016;
                if (pipelineWaveTime >= WAVE_INTERVAL && isDataFlowing) {
                    pipelineWaveTime = 0;
                    createPipelineWave();
                }
                
                // Update pipeline waves (only if data flowing)
                if (isDataFlowing) {
                    pipelineWaves.forEach(wave => {
                        // Update all particles in this wave
                        wave.particles.forEach(p => {
                            drawParticle(p);
                            p.progress += p.speed;
                            
                            // When particle reaches destination, spawn next stage particles
                            if (p.progress >= 1 && !p.spawned) {
                                p.spawned = true;
                                p.progress = 1;
                                
                                // Find next stage connections from this node
                                const nextLayer = p.conn.layer + 1;
                                if (nextLayer <= 7) {
                                    connections.filter(c => c.layer === nextLayer && c.from === p.conn.to).forEach(nextConn => {
                                        const newP = createParticle(nextConn, wave.id);
                                        newP.waveStage = nextLayer;
                                        newP.size = 5;  // Larger as data accumulates
                                        newP.color = nodes[nextConn.from].color;
                                        wave.particles.push({ ...newP, conn: nextConn });
                                    });
                                }
                            }
                        });
                        
                        // Clean up completed particles (but keep wave for a bit)
                        wave.particles = wave.particles.filter(p => p.progress < 1.2);
                    });
                }
                
                // Remove empty waves
                pipelineWaves = pipelineWaves.filter(w => w.particles.length > 0 || (Date.now() - w.startTime) < 5000);
                
                // Draw background flow particles (only subtle flow if data not flowing)
                dataParticles.forEach(p => {
                    if (p.isBackgroundFlow) {
                        ctx.globalAlpha = isDataFlowing ? 0.4 : 0.15;
                        drawParticle(p);
                        ctx.globalAlpha = 1;
                    } else if (isDataFlowing) {
                        drawParticle(p);
                    }
                    if (isDataFlowing) {
                        p.progress += p.speed * 0.7;
                        if (p.progress >= 1) {
                            p.progress = 0;
                            p.color = nodes[p.conn.to].color;
                        }
                    }
                });
                
                // Draw title with neon glow effect
                ctx.font = 'bold 12px Orbitron, sans-serif';
                ctx.fillStyle = '#00d4ff';
                ctx.shadowColor = '#00d4ff';
                ctx.shadowBlur = 15;
                ctx.textAlign = 'left';
                ctx.fillText('DATA PIPELINE: Bybit → Scan → PhD Math → Gemini AI → Trade', 15, 20);
                ctx.shadowBlur = 0;
                
                // Mini status with glow - shows actual state
                ctx.textAlign = 'right';
                ctx.font = 'bold 11px Rajdhani, sans-serif';
                const statusPulse = Math.sin(time * 4) * 0.3 + 0.7;
                
                // Determine actual status
                let statusText = '○ IDLE';
                let statusColor = '#888';
                let statusGlow = '#666';
                
                if (!isDataFlowing) {
                    // Check for specific errors
                    const hasMarketError = window.machineNodeErrors && window.machineNodeErrors.bybit;
                    const hasIndicatorError = window.machineNodeErrors && window.machineNodeErrors.indicators;
                    if (hasMarketError) {
                        statusText = '✕ DISCONNECTED';
                        statusColor = '#ff4444';
                        statusGlow = '#ff4444';
                    } else if (hasIndicatorError) {
                        statusText = '⚠ NO DATA';
                        statusColor = '#ffcc00';
                        statusGlow = '#ffcc00';
                    } else {
                        statusText = '○ WAITING';
                        statusColor = '#888';
                        statusGlow = '#666';
                    }
                } else if (position) {
                    statusText = '⚡ POSITION ACTIVE';
                    statusColor = `rgba(0, 255, 136, ${statusPulse})`;
                    statusGlow = '#00ff88';
                } else {
                    statusText = '● SCANNING';
                    statusColor = '#00d4ff';
                    statusGlow = '#00d4ff';
                }
                
                ctx.fillStyle = statusColor;
                ctx.shadowColor = statusGlow;
                ctx.shadowBlur = position ? 12 + statusPulse * 8 : 8;
                ctx.fillText(statusText, W - 15, 20);
                ctx.shadowBlur = 0;
                
                // Add processing indicator (only animate when data flowing)
                const procX = W - 15;
                const procY = 35;
                const procSpeed = position ? 6 : 3;
                for (let i = 0; i < 5; i++) {
                    const alpha = isDataFlowing ? (Math.sin(time * procSpeed + i * 0.5) + 1) / 2 : 0.2;
                    ctx.fillStyle = isDataFlowing ? `rgba(0, 212, 255, ${alpha * 0.8})` : `rgba(100, 100, 100, ${alpha})`;
                    ctx.beginPath();
                    ctx.arc(procX - i * 8, procY, 2, 0, Math.PI * 2);
                    ctx.fill();
                }
                
                // ═══════════════════════════════════════════════════════════════
                // 🌊 UPDATE FLOW INDICATOR & MACHINE GLOW
                // ═══════════════════════════════════════════════════════════════
                const flowIndicator = document.getElementById('pipeline-flow-indicator');
                const flowDot = document.getElementById('flow-dot');
                const flowText = document.getElementById('flow-status-text');
                const machineSection = document.querySelector('.machine-section');
                
                if (flowIndicator && flowDot && flowText) {
                    const positions = data.open_position || data.positions || [];
                    const hasPos = positions.length > 0 || (positions && positions.symbol);
                    const side = hasPos ? (positions[0]?.side || positions.side || 'LONG') : null;
                    const isLong = side === 'LONG';
                    
                    // Always update - remove all state classes first
                    flowIndicator.classList.remove('position-active', 'position-short');
                    flowDot.classList.remove('scanning', 'short');
                    
                    if (hasPos) {
                        // Position active - show flowing with energy
                        flowIndicator.classList.add(isLong ? 'position-active' : 'position-short');
                        flowDot.classList.toggle('short', !isLong);
                        flowText.textContent = isLong ? '🟢 LONG ACTIVE • ENERGY FLOWING' : '🔴 SHORT ACTIVE • ENERGY FLOWING';
                        flowText.style.color = isLong ? '#00ff88' : '#ff4444';
                        
                        // Machine section glow
                        if (machineSection) {
                            machineSection.classList.remove('position-short-active', 'position-active');
                            machineSection.classList.add(isLong ? 'position-active' : 'position-short-active');
                        }
                    } else if (isDataFlowing) {
                        // Scanning mode
                        flowDot.classList.add('scanning');
                        flowText.textContent = '📡 SCANNING FOR SIGNALS...';
                        flowText.style.color = '#00d4ff';
                        
                        if (machineSection) {
                            machineSection.classList.remove('position-active', 'position-short-active');
                        }
                    } else {
                        // Idle/Disconnected
                        flowText.textContent = '⏸️ STANDBY';
                        flowText.style.color = '#666';
                        
                        if (machineSection) {
                            machineSection.classList.remove('position-active', 'position-short-active');
                        }
                    }
                }
                
              } catch(e) {
                console.error('Machine animation error:', e);
              }
                machineAnimationId = requestAnimationFrame(animate);
            }
            
            // Cancel previous animation if running
            if (machineAnimationId) {
                cancelAnimationFrame(machineAnimationId);
            }
            
            animate();
        }
        
        // ============================================================
        // MACHINE LIVE EVENT FEED SYSTEM
        // ============================================================
        var machineEvents = [];
        var machineLastState = {};
        var machineNodeStates = {}; // Track active/error states for nodes
        
        function addMachineEvent(icon, message, type) {
            // type: 'info', 'success', 'warning', 'error', 'trade'
            var colors = {
                'info': '#00d4ff',
                'success': '#00ff88',
                'warning': '#ffcc00',
                'error': '#ff4444',
                'trade': '#aa00ff'
            };
            var now = new Date();
            var time = now.toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'});
            
            machineEvents.unshift({
                time: time,
                icon: icon,
                message: message,
                color: colors[type] || '#888'
            });
            
            // Keep last 50 events
            if (machineEvents.length > 50) machineEvents.pop();
            
            // Update display
            var container = document.getElementById('machine-events');
            if (container) {
                var html = machineEvents.map(function(e) {
                    return '<div style="padding: 4px 6px; margin: 2px 0; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 2px solid ' + e.color + ';">' +
                        '<span style="color: #666; font-size: 0.65rem;">' + e.time + '</span> ' +
                        '<span>' + e.icon + '</span> ' +
                        '<span style="color: ' + e.color + ';">' + e.message + '</span>' +
                    '</div>';
                }).join('');
                container.innerHTML = html;
            }
            
            // Flash the status dot
            var dot = document.getElementById('machine-status-dot');
            if (dot) {
                dot.style.background = colors[type] || '#00ff88';
                setTimeout(function() { dot.style.background = '#00ff88'; }, 500);
            }
        }
        
        function highlightMachineNode(nodeKey, color, duration) {
            // Store highlight state for the animation to pick up
            machineNodeStates[nodeKey] = { color: color, until: Date.now() + (duration || 2000) };
        }
        
        // Set persistent error state on a node
        function setMachineNodeError(nodeKey, hasError) {
            if (!window.machineNodeErrors) window.machineNodeErrors = {};
            window.machineNodeErrors[nodeKey] = hasError;
            if (hasError) {
                // Permanent error highlight until fixed
                machineNodeStates[nodeKey] = { color: '#ff4444', until: Date.now() + 60000, isError: true };
            } else if (window.machineNodeErrors[nodeKey] === false) {
                // Clear error state
                if (machineNodeStates[nodeKey] && machineNodeStates[nodeKey].isError) {
                    delete machineNodeStates[nodeKey];
                }
            }
        }
        
        // Set warning state on a node
        function setMachineNodeWarning(nodeKey, hasWarning) {
            if (!window.machineNodeWarnings) window.machineNodeWarnings = {};
            window.machineNodeWarnings[nodeKey] = hasWarning;
            if (hasWarning) {
                machineNodeStates[nodeKey] = { color: '#ffcc00', until: Date.now() + 60000, isWarning: true };
            } else if (window.machineNodeWarnings[nodeKey] === false) {
                if (machineNodeStates[nodeKey] && machineNodeStates[nodeKey].isWarning) {
                    delete machineNodeStates[nodeKey];
                }
            }
        }
        
        // Initialize global state
        window.machineDataFlowing = false;
        window.machineNodeErrors = {};
        window.machineNodeWarnings = {};
        
        function updateMachineFromData(data) {
            if (!data) {
                addMachineEvent('❌', 'No data from API', 'error');
                setMachineNodeError('bybit', true);
                setMachineNodeError('ohlcv', true);
                window.machineDataFlowing = false;
                return;
            }
            
            var prev = machineLastState;
            
            // === CHECK COMPONENT HEALTH ===
            var status = data.status || {};
            var indicators = data.indicators || {};
            var regime = data.regime || {};
            var ai = data.ai || {};
            var position = data.open_position || {};
            var risk = data.risk || {};
            
            // Bybit Exchange connectivity
            var isConnected = status.connected !== false;
            setMachineNodeError('bybit', !isConnected);
            setMachineNodeError('ohlcv', !isConnected);
            
            // Indicators - check if we have valid data
            var hasIndicators = indicators.rsi !== undefined || indicators.adx !== undefined;
            setMachineNodeError('indicators', !hasIndicators && isConnected);
            
            // Regime detection
            var hasRegime = regime.regime && regime.regime !== 'unknown';
            setMachineNodeError('regime', !hasRegime && hasIndicators);
            
            // Scanner
            var scannerData = data.scanner || {};
            setMachineNodeError('scanner', false);
            
            // PhD Math
            setMachineNodeError('math', false);
            
            // Gemini AI 
            var aiAvailable = ai.total_signals !== undefined || ai.approved !== undefined;
            setMachineNodeError('gemini', false);
            
            // Risk manager
            var riskOk = risk.can_trade !== false;
            setMachineNodeWarning('risk', !riskOk);
            
            // Execution
            setMachineNodeError('execute', false);
            
            // Position monitor
            setMachineNodeError('position', false);
            
            // Telegram
            var pipelineCompsForTele = (window.lastPipelineData && window.lastPipelineData.components) || {};
            var teleComp = pipelineCompsForTele.telegram;
            var teleConnected = teleComp && (teleComp.connected || teleComp.is_started);
            var teleHasError = teleComp && teleComp.status === 'error';
            var teleHasWarning = teleComp && teleComp.status === 'warning' && !teleConnected;
            setMachineNodeError('telegram', teleHasError);
            setMachineNodeWarning('telegram', teleHasWarning);
            if (teleConnected && !teleHasError && !teleHasWarning) {
                setMachineNodeError('telegram', false);
                setMachineNodeWarning('telegram', false);
            }
            
            // Set data flowing state
            window.machineDataFlowing = isConnected && hasIndicators;
            
            // === UPDATE NODE SUB-LABELS WITH LIVE DATA ===
            if (window.machineNodes) {
                var n = window.machineNodes;
                var pipelineComps = (window.lastPipelineData && window.lastPipelineData.components) || {};
                
                // BYBIT - show price
                var marketPrice = (pipelineComps.market_feed && pipelineComps.market_feed.last_price) || 
                                  status.current_price;
                n.bybit.sub = marketPrice ? '$' + parseFloat(marketPrice).toFixed(1) : (isConnected ? 'OK' : 'ERR');
                
                // OHLCV - show bars loaded
                var barsLoaded = pipelineComps.market_feed && pipelineComps.market_feed.bars_loaded;
                n.ohlcv.sub = isConnected ? (barsLoaded ? barsLoaded + ' bars' : '15m') : 'OFF';
                
                // Indicators - show RSI 
                var indRsi = (pipelineComps.indicators && pipelineComps.indicators.rsi) || indicators.rsi;
                n.indicators.sub = indRsi ? 'RSI ' + Math.round(indRsi) : '--';
                
                // Regime - show current regime
                var regimeText = (pipelineComps.regime && pipelineComps.regime.current) || regime.regime || 'UNKNOWN';
                n.regime.sub = regimeText.substring(0, 4).toUpperCase();
                
                // Scanner - show pairs scanned
                var scanCount = (pipelineComps.scanner && pipelineComps.scanner.pairs_scanned) || 
                                (scannerData.pairs_scanned) || 50;
                n.scanner.sub = scanCount + ' PAIRS';
                
                // PhD Math - show score
                var mathScore = (pipelineComps.ai_filter && pipelineComps.ai_filter.last_score) ||
                               (ai && ai.last_score);
                n.math.sub = mathScore ? Math.round(mathScore) + '/100' : 'RDY';
                
                // Gemini AI - show mode or approval status
                var aiMode = (pipelineComps.ai_filter && pipelineComps.ai_filter.mode) || (ai && ai.mode);
                var aiCalls = (pipelineComps.ai_filter && pipelineComps.ai_filter.call_count);
                n.gemini.sub = aiMode ? aiMode.substring(0, 4).toUpperCase() : (aiCalls ? '#' + aiCalls : 'RDY');
                
                // Risk - show can_trade status
                var riskCanTrade = (pipelineComps.risk_manager && pipelineComps.risk_manager.can_trade) || riskOk;
                n.risk.sub = riskCanTrade ? 'OK' : 'STOP';
                
                // Execute - show last action
                var posMonitor = pipelineComps.position_monitor;
                var posStatus = (posMonitor && posMonitor.has_position) ? 'ACTIVE' : 'IDLE';
                n.execute.sub = posStatus;
                
                // Position - show side if active
                var posSide = (posMonitor && posMonitor.has_position) ? posMonitor.position_side.toUpperCase() : 
                              (position && position.side ? position.side.toUpperCase() : 'NIL');
                n.position.sub = posSide;
                
                // Telegram
                n.telegram.sub = (teleComp && (teleComp.connected || teleComp.is_started)) ? 'ON' : 'OFF';
            }
            
            // === DETECT STATE CHANGES ===
            
            // 1. Position opened/closed
            var hasPosition = data.open_position && data.open_position.side;
            var hadPosition = prev.hasPosition;
            if (hasPosition && !hadPosition) {
                var side = data.open_position.side.toUpperCase();
                var symbol = data.open_position.symbol || data.status?.symbol || '???';
                var entry = data.open_position.entry ? '$' + data.open_position.entry.toFixed(2) : '';
                addMachineEvent('💹', side + ' ' + symbol + ' @ ' + entry, 'trade');
                highlightMachineNode('execute', '#00ff88', 3000);
                highlightMachineNode('position', '#00ff88', 3000);
            } else if (!hasPosition && hadPosition) {
                var pnl = prev.lastPnl || 0;
                var icon = pnl >= 0 ? '✅' : '❌';
                var type = pnl >= 0 ? 'success' : 'error';
                addMachineEvent(icon, 'Trade closed: ' + (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%', type);
                highlightMachineNode('position', pnl >= 0 ? '#00ff88' : '#ff4444', 3000);
            }
            
            // 2. AI activity
            var ai = data.ai || {};
            var aiTotal = ai.total_signals || 0;
            var prevAiTotal = prev.aiTotal || 0;
            if (aiTotal > prevAiTotal) {
                addMachineEvent('🤖', 'Gemini evaluated ' + (aiTotal - prevAiTotal) + ' signal(s)', 'info');
                highlightMachineNode('gemini', '#4285f4', 1500);
                highlightMachineNode('math', '#00ff88', 1500);
            }
            
            // 3. Regime changes
            var regime = data.regime || {};
            var regimeName = regime.regime || 'unknown';
            var prevRegime = prev.regimeName || 'unknown';
            if (regimeName !== prevRegime && regimeName !== 'unknown') {
                var regimeIcon = regimeName === 'trending' ? '📈' : regimeName === 'ranging' ? '📊' : '🌊';
                addMachineEvent(regimeIcon, 'Regime: ' + regimeName.toUpperCase(), 'warning');
                highlightMachineNode('regime', '#ffcc00', 2000);
            }
            
            // 4. Scanner activity
            var scanPairs = (data.scanner && data.scanner.pairs_scanned) || 0;
            var prevScanPairs = prev.scanPairs || 0;
            if (scanPairs > 0 && scanPairs !== prevScanPairs) {
                addMachineEvent('🔬', 'Scanned ' + scanPairs + ' pairs', 'info');
                highlightMachineNode('scanner', '#aa00ff', 1500);
            }
            
            // 5. Symbol switch
            var symbol = data.status?.symbol || '';
            var prevSymbol = prev.symbol || '';
            if (symbol !== prevSymbol && prevSymbol !== '') {
                addMachineEvent('🔄', 'Switched to ' + symbol, 'warning');
                highlightMachineNode('bybit', '#f7a600', 2000);
                highlightMachineNode('ohlcv', '#00d4ff', 2000);
            }
            
            // 6. Errors from status
            var errors = data.errors || [];
            var prevErrorCount = prev.errorCount || 0;
            if (errors.length > prevErrorCount) {
                var newErrors = errors.slice(prevErrorCount);
                newErrors.forEach(function(err) {
                    addMachineEvent('⚠️', err.substring(0, 40) + '...', 'error');
                });
            }
            
            // Store current state
            machineLastState = {
                hasPosition: hasPosition,
                lastPnl: data.open_position?.pnl_percent || 0,
                aiTotal: aiTotal,
                regimeName: regimeName,
                scanPairs: scanPairs,
                symbol: symbol,
                errorCount: errors.length
            };
            
            // Trigger a pipeline data wave on significant state changes
            if (typeof window.createPipelineWave === 'function') {
                var shouldTriggerWave = (hasPosition !== prev.hasPosition) || 
                                        (aiTotal > prevAiTotal);
                if (shouldTriggerWave) {
                    window.createPipelineWave();
                }
            }
        }
        
        // Hook into updateDashboard to track events
        var originalUpdateDashboard = updateDashboard;
        updateDashboard = function(data) {
            originalUpdateDashboard(data);
            updateMachineFromData(data);
        };
        
        // Initial events on load
        setTimeout(function() {
            addMachineEvent('🚀', 'Dashboard connected', 'success');
            addMachineEvent('📊', 'Fetching market data...', 'info');
        }, 1500);
        
        // Auto-start machine animation when page loads
        setTimeout(initMachineAnimation, 1000);
        
        // Close popup on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                const popup = document.querySelector('.pipeline-popup.active');
                if (popup) popup.remove();
                currentOpenPopup = null;
            }
        });
    </script>
</body>
</html>
"""



class Dashboard:
    """
    Web-based Performance Dashboard for Julaba.
    
    Provides real-time monitoring of:
    - Account balance and P&L
    - Open positions
    - Trade history
    - Market regime
    - AI filter statistics
    - Technical indicators
    - Risk management
    - Multi-timeframe analysis
    - ML model status
    - System logs
    """
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.app = None
        self.server_thread = None
        self.running = False
        
        # Data callbacks (set by main bot)
        self.get_status = None
        self.get_balance = None
        self.get_pnl = None
        self.get_position = None
        self.get_additional_positions = None  # NEW: For multi-pair positions
        self.get_trades = None
        self.get_regime = None
        self.get_ai_stats = None
        self.get_equity_curve = None
        # Enhanced callbacks
        self.get_indicators = None
        self.get_current_signal = None
        self.get_risk_stats = None
        self.get_mtf_analysis = None
        self.get_params = None
        self.get_signals = None
        self.get_ohlc_data = None  # For live price chart
        # NEW: ML and logs callbacks
        self.get_ml_status = None
        self.get_system_logs = None
        # AI explanation callback
        self.get_ai_explanation = None
        # Market scanner callbacks
        self.get_market_scan = None
        self.switch_symbol = None
        self.ai_analyze_markets = None
        # NEW: AI tracker and pre-filter stats
        self.get_ai_tracker_stats = None
        self.get_prefilter_stats = None
        # Full state callback
        self.get_full_state = None
        
        # === NEW: PIPELINE MONITORING ===
        self.get_pipeline_status = None  # Real-time pipeline health
        
        # === NEW: CONTROL PANEL CALLBACKS ===
        self.do_pause = None
        self.do_resume = None
        self.do_stop = None
        self.do_close_position = None
        self.toggle_position_hold = None  # NEW: Hold toggle callback
        self.do_open_trade = None  # NEW: Manual trade execution
        self.get_adaptive_params = None
        self.set_adaptive_param = None
        self.set_system_param = None  # For force_resume, etc.
        self.chat_with_ai = None
        self.trigger_auto_tune = None
        self.set_ai_mode = None  # NEW: Set AI trading mode callback
        self.get_error_summary = None  # NEW: Get error history for display
        self.clear_errors = None  # NEW: Clear error history
        
        if FLASK_AVAILABLE:
            self._setup_flask()
    
    
    
    
    
    
    
    
    
    
    
    def _setup_flask(self):
        """Setup Flask application."""
        self.app = Flask(__name__)
        
        # Use custom JSON provider to handle numpy types
        if FLASK_AVAILABLE:
            self.app.json_provider_class = NumpyJSONProvider
            self.app.json = NumpyJSONProvider(self.app)
        
        self.app.logger.setLevel(logging.WARNING)
        
        # Suppress Flask logs
        import logging as flask_logging
        flask_logging.getLogger('werkzeug').setLevel(flask_logging.WARNING)
        
        # Add CORS headers to all responses (restricted to same-origin)
        @self.app.after_request
        def add_cors_headers(response):
            # Only allow same-origin requests (no wildcard)
            origin = request.headers.get('Origin', '')
            if origin and ('localhost' in origin or '127.0.0.1' in origin):
                response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        @self.app.route('/debug')
        def debug_page():
            """Ultra-minimal debug page."""
            return '''<!DOCTYPE html>
<html>
<head><title>Debug</title></head>
<body style="background:#000;color:#0f0;font-family:monospace;padding:20px;font-size:14px;">
<h1>JULABA DEBUG PAGE</h1>
<div id="log" style="white-space:pre-wrap;"></div>
<script>
function log(msg) {
    document.getElementById('log').textContent += new Date().toISOString() + ' - ' + msg + '\n';
    console.log(msg);
}
log('Script started');
log('Window location: ' + window.location.href);
log('Protocol: ' + window.location.protocol);
log('Attempting fetch to /api/data...');
var xhr = new XMLHttpRequest();
xhr.open('GET', '/api/data', true);
xhr.onreadystatechange = function() {
    log('XHR state: ' + xhr.readyState + ', status: ' + xhr.status);
    if (xhr.readyState === 4) {
        if (xhr.status === 200) {
            log('SUCCESS! Response length: ' + xhr.responseText.length);
            try {
                var data = JSON.parse(xhr.responseText);
                log('Parsed JSON successfully');
                log('Balance: $' + (data.balance ? data.balance.current : 'N/A'));
                log('Price: $' + (data.status ? data.status.current_price : 'N/A'));
                log('Symbol: ' + (data.status ? data.status.symbol : 'N/A'));
            } catch(e) {
                log('JSON parse error: ' + e.message);
            }
        } else {
            log('HTTP Error: ' + xhr.status + ' ' + xhr.statusText);
        }
    }
};
xhr.onerror = function() {
    log('XHR ERROR: Network error occurred');
};
xhr.send();
log('XHR request sent');
</script>
</body>
</html>'''

        @self.app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        
        @self.app.route('/api/data')
        def api_data():
            data = {}
            
            try:
                # NEW: Enhanced data
                if self.get_status:
                    data['status'] = self.get_status()
                if self.get_balance:
                    data['balance'] = self.get_balance()
                if self.get_pnl:
                    data['pnl'] = self.get_pnl()
                # Position data - use primary position as open_position for dashboard
                if self.get_position:
                    data['open_position'] = self.get_position()
                # Additional positions for multi-pair
                if self.get_additional_positions:
                    data['additional_positions'] = self.get_additional_positions()
                else:
                    data['additional_positions'] = []
                if self.get_trades:
                    data['trades'] = self.get_trades()
                if self.get_regime:
                    data['regime'] = self.get_regime()
                if self.get_ai_stats:
                    data['ai'] = self.get_ai_stats()
                if self.get_equity_curve:
                    data['equity_curve'] = self.get_equity_curve()
                # NEW: Enhanced data
                if self.get_indicators:
                    data['indicators'] = self.get_indicators()
                if self.get_current_signal:
                    data['current_signal'] = self.get_current_signal()
                if self.get_risk_stats:
                    data['risk'] = self.get_risk_stats()
                if self.get_mtf_analysis:
                    data['mtf'] = self.get_mtf_analysis()
                if self.get_params:
                    data['params'] = self.get_params()
                if self.get_signals:
                    data['signals'] = self.get_signals()
                # REMOVED: OHLC from /api/data - use dedicated /api/ohlc endpoint instead
                # This was causing excessive API calls (every 2 sec instead of using cache)
                # if self.get_ohlc_data:
                #     data['ohlc'] = self.get_ohlc_data()
                # NEW: ML and logs
                if self.get_ml_status:
                    data['ml'] = self.get_ml_status()
                if self.get_system_logs:
                    data['logs'] = self.get_system_logs()
                # NEW: AI tracker and pre-filter stats
                if self.get_ai_tracker_stats:
                    data['ai_tracker'] = self.get_ai_tracker_stats()
                if self.get_prefilter_stats:
                    data['prefilter'] = self.get_prefilter_stats()
            except Exception as e:
                logger.error(f"Dashboard API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/state')
        def api_state():
            """Unified system state - SINGLE SOURCE OF TRUTH for all components."""
            data = {}
            try:
                if self.get_full_state:
                    data = self.get_full_state()
                else:
                    data['error'] = 'Full state not available'
            except Exception as e:
                logger.error(f"Dashboard state API error: {e}")
                data['error'] = str(e)
            return jsonify(data)
        
        @self.app.route('/api/pipeline')
        def api_pipeline():
            """Real-time pipeline status - shows data flow and component health."""
            data = {
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'data_flow': [],
                'errors': [],
                'warnings': []
            }
            try:
                if self.get_pipeline_status:
                    pipeline_data = self.get_pipeline_status()
                    data.update(pipeline_data)
                else:
                    # Provide basic status from available callbacks
                    data['components'] = {
                        'market_feed': {'status': 'ok' if self.get_ohlc_data else 'unknown'},
                        'indicators': {'status': 'ok' if self.get_indicators else 'unknown'},
                        'ai_filter': {'status': 'ok' if self.get_ai_stats else 'unknown'},
                        'risk_manager': {'status': 'ok' if self.get_risk_stats else 'unknown'}
                    }
            except Exception as e:
                logger.error(f"Dashboard pipeline API error: {e}")
                data['error'] = str(e)
            return jsonify(data)
        
        # OHLC cache to prevent excessive exchange API calls
        # Key: (symbol, tf, range) -> {'data': candles, 'timestamp': time}
        self._ohlc_cache = {}
        OHLC_CACHE_TTL = 30  # Cache for 30 seconds (reduced API calls)
        
        @self.app.route('/api/ohlc')
        def api_ohlc():
            """Get OHLC candlestick data for live chart with date range support."""
            import time as _time
            data = {'candles': []}
            tf = request.args.get('tf', '1m')
            date_range = request.args.get('range', '1d')
            date_from = request.args.get('from', None)
            date_to = request.args.get('to', None)
            symbol = request.args.get('symbol', None)  # Optional: for Position 2 chart
            
            # Check cache first (only for standard range requests, not custom dates)
            cache_key = (symbol or 'default', tf, date_range)
            now = _time.time()
            
            if not date_from and not date_to and cache_key in self._ohlc_cache:
                cached = self._ohlc_cache[cache_key]
                if now - cached['timestamp'] < OHLC_CACHE_TTL:
                    # Return cached data
                    data['candles'] = cached['data']
                    data['timeframe'] = tf
                    data['range'] = date_range
                    data['cached'] = True
                    if symbol:
                        data['symbol'] = symbol
                    return jsonify(data)
            
            logger.info(f"OHLC API called: tf={tf}, range={date_range}, symbol={symbol}, callback={self.get_ohlc_data is not None}")
            
            try:
                if self.get_ohlc_data:
                    # Pass range params to fetcher, include symbol if specified
                    candles = self.get_ohlc_data(tf, date_range, date_from, date_to, symbol)
                    logger.info(f"OHLC API: Got {len(candles) if candles else 0} candles")
                    data['candles'] = candles
                    data['timeframe'] = tf
                    data['range'] = date_range
                    if symbol:
                        data['symbol'] = symbol
                    
                    # Cache the result (only for standard range requests)
                    if not date_from and not date_to and candles:
                        self._ohlc_cache[cache_key] = {
                            'data': candles,
                            'timestamp': now
                        }
                        # Clean old cache entries (keep max 10)
                        if len(self._ohlc_cache) > 10:
                            oldest_key = min(self._ohlc_cache, key=lambda k: self._ohlc_cache[k]['timestamp'])
                            del self._ohlc_cache[oldest_key]
                else:
                    logger.warning("OHLC API: get_ohlc_data callback is None!")
            except Exception as e:
                logger.error(f"Dashboard OHLC API error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/logs')
        def api_logs():
            """Get system logs for dashboard."""
            data = {'logs': []}
            count = request.args.get('count', 50, type=int)
            
            try:
                if self.get_system_logs:
                    data['logs'] = self.get_system_logs(count)
            except Exception as e:
                logger.error(f"Dashboard logs API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/ai-explain', methods=['POST'])
        def api_ai_explain():
            """Get AI explanation for a dashboard topic."""
            data = {}
            
            try:
                req_data = request.get_json() or {}
                topic = req_data.get('topic', '')
                display_name = req_data.get('display_name', topic)
                # Check topic
                if not topic:
                    return jsonify({'error': 'No topic specified'})
                
                if self.get_ai_explanation:
                    explanation = self.get_ai_explanation(topic, display_name)
                    data['explanation'] = explanation
                    data['topic'] = topic
                else:
                    data['error'] = 'AI explanation service not available'
            except Exception as e:
                logger.error(f"Dashboard AI explain API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/market-scan')
        def api_market_scan():
            """Get multi-pair market data for scanner."""
            data = {'pairs': [], 'current_symbol': '', 'multi_pair_enabled': False, 'multi_pair_count': 0, 'symbols_with_positions': [], 'position_count': 0}
            
            try:
                if self.get_market_scan:
                    scan_data = self.get_market_scan()
                    data['pairs'] = scan_data.get('pairs', [])
                    data['current_symbol'] = scan_data.get('current_symbol', '')
                    data['multi_pair_enabled'] = scan_data.get('multi_pair_enabled', False)
                    data['multi_pair_count'] = scan_data.get('multi_pair_count', 0)
                    data['active_pairs'] = scan_data.get('active_pairs', [])
                    # Include position info for scanner highlighting
                    data['symbols_with_positions'] = scan_data.get('symbols_with_positions', [])
                    data['position_count'] = scan_data.get('position_count', 0)
                    # Dynamic scan info
                    data['scan_pair_count'] = scan_data.get('scan_pair_count', 0)
                    data['scan_source'] = scan_data.get('scan_source', 'unknown')
            except Exception as e:
                logger.error(f"Dashboard market scan API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/switch-symbol', methods=['POST'])
        def api_switch_symbol():
            """Switch trading symbol."""
            data = {}
            
            try:
                req_data = request.get_json() or {}
                symbol = req_data.get('symbol', '')
                
                # Check symbol
                if not symbol:
                    return jsonify({'error': 'No symbol specified', 'success': False})
                if self.switch_symbol:
                    result = self.switch_symbol(symbol)
                    data['success'] = result.get('success', False)
                    data['message'] = result.get('message', '')
                    if not result.get('success'):
                        data['error'] = result.get('error', 'Unknown error')
                else:
                    data['error'] = 'Symbol switch not available'
                    data['success'] = False
            except Exception as e:
                logger.error(f"Dashboard switch symbol API error: {e}")
                data['error'] = str(e)
                data['success'] = False
            
            return jsonify(data)
        
        @self.app.route('/api/ai-analyze-markets', methods=['POST'])
        def api_ai_analyze_markets():
            """Get AI analysis of all market pairs."""
            data = {}
            
            try:
                if self.ai_analyze_markets:
                    result = self.ai_analyze_markets()
                    data['recommendation'] = result.get('recommendation', '')
                    data['best_pair'] = result.get('best_pair', '')
                else:
                    data['error'] = 'AI market analysis not available'
            except Exception as e:
                logger.error(f"Dashboard AI analyze markets API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        # ============================================================
        # CONTROL PANEL API ENDPOINTS
        # ============================================================
        
        @self.app.route('/api/errors')
        def api_errors():
            """Get error history for dashboard display."""
            try:
                if hasattr(self, 'get_error_summary') and self.get_error_summary:
                    return jsonify(self.get_error_summary())
                return jsonify({
                    'total_errors': 0,
                    'recent_errors': 0,
                    'last_error': None,
                    'last_error_time': None,
                    'error_history': [],
                    'engine_running': False
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/errors/clear', methods=['POST'])
        def api_clear_errors():
            """Clear error history."""
            try:
                if self.clear_errors:
                    result = self.clear_errors()
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Clear errors not available'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/control')
        def control_panel():
            """Control panel removed — redirect to dashboard."""
            return redirect('/')
        

        
        # ==================== NEW FEATURES API ROUTES ====================
        
        @self.app.route('/api/portfolio')
        def api_portfolio():
            """Get comprehensive portfolio analytics."""
            try:
                from portfolio_manager import PortfolioManager
                pm = PortfolioManager()
                pm.reload_trades()
                return jsonify(pm.get_full_report())
            except Exception as e:
                logger.error(f"Portfolio API error: {e}")
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/portfolio/daily')
        def api_portfolio_daily():
            """Get daily P&L breakdown."""
            try:
                from portfolio_manager import PortfolioManager
                pm = PortfolioManager()
                pm.reload_trades()
                days = request.args.get('days', 30, type=int)
                return jsonify([asdict(d) for d in pm.get_daily_pnl(days)])
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/portfolio/by-pair')
        def api_portfolio_by_pair():
            """Get performance breakdown by trading pair."""
            try:
                from portfolio_manager import PortfolioManager
                from dataclasses import asdict
                pm = PortfolioManager()
                pm.reload_trades()
                stats = pm.get_stats_by_pair()
                return jsonify({pair: asdict(m) for pair, m in stats.items()})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/portfolio/by-direction')
        def api_portfolio_by_direction():
            """Get performance breakdown by direction (LONG/SHORT)."""
            try:
                from portfolio_manager import PortfolioManager
                from dataclasses import asdict
                pm = PortfolioManager()
                pm.reload_trades()
                stats = pm.get_stats_by_direction()
                return jsonify({d: asdict(m) for d, m in stats.items()})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """Get all alerts and recent history."""
            try:
                from alert_manager import AlertManager
                am = AlertManager()
                return jsonify(am.get_summary())
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/alerts/add', methods=['POST'])
        def api_alerts_add():
            """Add a new alert."""
            try:
                from alert_manager import AlertManager, AlertType, AlertPriority
                am = AlertManager()
                data = request.get_json() or {}
                
                alert_type = AlertType(data.get('type', 'price_above'))
                symbol = data.get('symbol', '')
                value = data.get('value', 0)
                message = data.get('message', '')
                priority = AlertPriority(data.get('priority', 'medium'))
                one_time = data.get('one_time', True)
                
                alert_id = am.add_alert(alert_type, symbol, value, message, priority, one_time)
                return jsonify({'success': True, 'alert_id': alert_id})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/alerts/remove', methods=['POST'])
        def api_alerts_remove():
            """Remove an alert."""
            try:
                from alert_manager import AlertManager
                am = AlertManager()
                data = request.get_json() or {}
                alert_id = data.get('alert_id', '')
                success = am.remove_alert(alert_id)
                return jsonify({'success': success})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/exchanges')
        def api_exchanges():
            """Get multi-exchange status."""
            try:
                from multi_exchange import MultiExchangeManager
                mem = MultiExchangeManager()
                return jsonify(mem.get_status())
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/backtest/monte-carlo', methods=['POST'])
        def api_monte_carlo():
            """Run Monte Carlo simulation."""
            try:
                from monte_carlo import MonteCarloSimulator
                data = request.get_json() or {}
                
                mc = MonteCarloSimulator(
                    initial_balance=data.get('initial_balance', 350),
                    risk_per_trade=data.get('risk_per_trade', 0.02)
                )
                
                result = mc.simulate(
                    n_simulations=data.get('simulations', 1000),
                    n_trades=data.get('trades', 100),
                    win_rate=data.get('win_rate', 0.55),
                    avg_win_r=data.get('avg_win_r', 1.5),
                    avg_loss_r=data.get('avg_loss_r', 1.0)
                )
                
                # Don't return full equity curves (too large)
                return jsonify({
                    'n_simulations': result.n_simulations,
                    'median_final_balance': result.median_final_balance,
                    'percentile_5': result.percentile_5,
                    'percentile_95': result.percentile_95,
                    'prob_profit': result.prob_profit,
                    'prob_ruin': result.prob_ruin,
                    'expected_return': result.expected_return,
                    'max_drawdown_median': result.max_drawdown_median,
                    'sharpe_equivalent': result.expected_return / (result.return_std + 0.001)
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        
        # News cache to avoid excessive API calls
        self._news_cache = {'data': None, 'timestamp': 0}
        self._news_fetching = False  # Lock to prevent concurrent fetches
        NEWS_CACHE_TTL = 120  # Cache news for 2 minutes (RSS feeds don't update faster)
        
        @self.app.route('/api/news')
        def api_news():
            """Get market news and sentiment summary."""
            import time as _time
            
            # Check cache first
            now = _time.time()
            if self._news_cache['data'] and (now - self._news_cache['timestamp']) < NEWS_CACHE_TTL:
                return jsonify(self._news_cache['data'])
            
            # Prevent concurrent fetches (news fetch takes 5-10s, don't pile up)
            if self._news_fetching:
                if self._news_cache['data']:
                    return jsonify(self._news_cache['data'])
                return jsonify({'news_summary': {'total_articles': 0}, 'recent_news': [], 'sentiment': None, 'whale_alerts': [], 'recommendation': 'Loading...', 'social_sentiment': {}, 'liquidations': {}, 'critical_news': [], 'llm_analysis': {}})
            
            self._news_fetching = True
            try:
                import asyncio
                from news_monitor import NewsMonitor
                
                monitor = NewsMonitor()
                
                # Run async function in sync context with timeout
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    summary = loop.run_until_complete(
                        asyncio.wait_for(monitor.get_market_summary(), timeout=20)
                    )
                finally:
                    loop.close()
                
                # Cache the result
                self._news_cache = {'data': summary, 'timestamp': now}
                
                return jsonify(summary)
            except asyncio.TimeoutError:
                logger.warning("News API timeout (20s) - returning cached data")
                if self._news_cache['data']:
                    return jsonify(self._news_cache['data'])
                return jsonify({'news_summary': {'total_articles': 0}, 'recent_news': [], 'sentiment': None, 'whale_alerts': [], 'recommendation': 'Loading...', 'social_sentiment': {}, 'liquidations': {}, 'critical_news': [], 'llm_analysis': {}})
            except Exception as e:
                logger.error(f"News API error: {e}")
                # Return cached data if available, even if stale
                if self._news_cache['data']:
                    return jsonify(self._news_cache['data'])
                return jsonify({'error': str(e)})
            finally:
                self._news_fetching = False
    
    def _get_bind_host(self) -> str:
        """Dashboard always binds to localhost for security."""
        return '127.0.0.1'
    
    def start(self):
        """Start the dashboard server in a background thread."""
        if not FLASK_AVAILABLE:
            logger.warning("Cannot start dashboard - Flask not installed")
            return False
        
        if self.running:
            return True
        
        bind_host = self._get_bind_host()
        
        def run_server():
            try:
                self.app.run(host=bind_host, port=self.port, debug=False, use_reloader=False, threaded=True)
            except Exception as e:
                logger.error(f"🚨 Dashboard server crashed: {e}")
                self.running = False
                # Auto-restart after 5 seconds
                import time
                time.sleep(5)
                logger.info("🔄 Dashboard auto-restarting...")
                try:
                    self.app.run(host=bind_host, port=self.port, debug=False, use_reloader=False, threaded=True)
                except Exception as e2:
                    logger.error(f"🚨 Dashboard restart also failed: {e2}")
                    self.running = False
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        
        if bind_host == '127.0.0.1':
            logger.info(f"📊 Dashboard started at http://localhost:{self.port} (local only)")
        else:
            logger.info(f"📊 Dashboard started at http://localhost:{self.port} (public)")
        return True
    
    def stop(self):
        """Stop the dashboard server."""
        self.running = False
        logger.info("Dashboard stopped")


# Singleton instance
_dashboard: Optional[Dashboard] = None


def get_dashboard(port: int = 5000) -> Dashboard:
    """Get the global dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = Dashboard(port=port)
    return _dashboard


# Standalone run for testing
if __name__ == "__main__":
    print("[INFO] Starting Julaba Dashboard main block...")
    if not FLASK_AVAILABLE:
        print("[ERROR] Flask is not installed. Run: pip install flask")
    else:
        dash = Dashboard(port=5000)
        if dash.app:
            print("[INFO] Flask app created. Running on port 5000...")
            dash.app.run(host="127.0.0.1", port=5000, debug=True)
        else:
            print("[ERROR] Dashboard app was not created.")
    
