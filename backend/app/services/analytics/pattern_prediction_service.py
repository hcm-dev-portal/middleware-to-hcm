# ================================================================================
# backend/app/services/analytics/pattern_prediction_service.py

from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics
from enum import Enum

