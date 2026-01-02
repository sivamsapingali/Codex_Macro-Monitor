"""
Sector Impact Prediction Model
Predicts 6-12 month sector impact based on macro indicator signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SECTOR_ETFS, SECTOR_SENSITIVITIES, REGIME_DEFINITIONS


def calculate_macro_regime(indicators_data: Dict) -> Dict:
    """
    Determine current macro regime based on indicator signals
    
    Args:
        indicators_data: Dict of {series_id: derivatives_summary}
    
    Returns:
        Regime classification with confidence
    """
    scores = {
        'early_cycle': 0,
        'mid_cycle': 0,
        'late_cycle': 0,
        'recession': 0
    }
    
    # Check for recession indicator directly
    if 'USREC' in indicators_data:
        if indicators_data['USREC'].get('value', 0) == 1:
            scores['recession'] += 3
    
    # Unemployment rate signals
    if 'UNRATE' in indicators_data:
        unrate_roc = indicators_data['UNRATE'].get('roc_3m', 0) or 0
        if unrate_roc < -0.2:  # Falling unemployment
            scores['early_cycle'] += 2
            scores['mid_cycle'] += 1
        elif unrate_roc > 0.3:  # Rising unemployment
            scores['recession'] += 2
            scores['late_cycle'] += 1
    
    # Industrial production signals
    if 'INDPRO' in indicators_data:
        indpro_roc = indicators_data['INDPRO'].get('roc_6m', 0) or 0
        indpro_accel = indicators_data['INDPRO'].get('acceleration', 0) or 0
        if indpro_roc > 2 and indpro_accel > 0:
            scores['early_cycle'] += 2
            scores['mid_cycle'] += 1
        elif indpro_roc > 0 and indpro_accel < 0:
            scores['late_cycle'] += 2
        elif indpro_roc < 0:
            scores['recession'] += 2
    
    # Yield curve signals
    if 'T10Y2Y' in indicators_data:
        spread = indicators_data['T10Y2Y'].get('value', 0) or 0
        if spread < 0:  # Inverted
            scores['late_cycle'] += 2
            scores['recession'] += 1
        elif spread > 1.5:  # Steep
            scores['early_cycle'] += 2
    
    # Fed funds direction
    if 'FEDFUNDS' in indicators_data:
        ff_roc = indicators_data['FEDFUNDS'].get('roc_6m', 0) or 0
        if ff_roc > 0.5:  # Rising rates
            scores['mid_cycle'] += 1
            scores['late_cycle'] += 1
        elif ff_roc < -0.5:  # Falling rates
            scores['recession'] += 1
            scores['early_cycle'] += 1
    
    # CPI signals
    if 'CPIAUCSL' in indicators_data:
        cpi_yoy = indicators_data['CPIAUCSL'].get('roc_12m', 0) or 0
        if cpi_yoy > 4:
            scores['late_cycle'] += 1
        elif cpi_yoy < 2:
            scores['recession'] += 1
    
    # Determine regime
    max_score = max(scores.values())
    total_score = sum(scores.values()) or 1
    
    regime = max(scores.keys(), key=lambda k: scores[k])
    confidence = (max_score / total_score) * 100 if total_score > 0 else 25
    
    return {
        'regime': regime,
        'confidence': round(confidence, 1),
        'scores': scores,
        'description': REGIME_DEFINITIONS[regime]['description'],
        'favored_sectors': REGIME_DEFINITIONS[regime]['favored_sectors'],
        'avoid_sectors': REGIME_DEFINITIONS[regime]['avoid_sectors']
    }


def calculate_sector_impact(
    sector_id: str,
    indicators_data: Dict,
    macro_regime: Dict,
    horizon_months: int = 9
) -> Dict:
    """
    Calculate predicted impact on a sector 6-12 months forward
    
    Args:
        sector_id: Sector ETF ticker (e.g., 'XLK')
        indicators_data: Dict of indicator derivatives
        macro_regime: Current regime classification
        horizon_months: Prediction horizon (6-12)
    
    Returns:
        Sector impact prediction with confidence and drivers
    """
    if sector_id not in SECTOR_SENSITIVITIES:
        return {'error': f'Unknown sector: {sector_id}'}
    
    sensitivities = SECTOR_SENSITIVITIES[sector_id]
    sector_info = SECTOR_ETFS[sector_id]
    
    # Calculate weighted impact score
    impact_score = 0
    drivers = []
    total_weight = 0
    
    for category, indicators in sensitivities.items():
        if category == 'description':
            continue
            
        for indicator_id, sensitivity in indicators.items():
            if indicator_id in indicators_data:
                data = indicators_data[indicator_id]
                
                # Use 6-month rate of change for forward-looking signal
                roc = data.get('roc_6m', 0) or 0
                accel = data.get('acceleration', 0) or 0
                
                # Impact = sensitivity * (rate of change + acceleration bonus)
                indicator_impact = sensitivity * (roc + accel * 0.5)
                impact_score += indicator_impact
                total_weight += abs(sensitivity)
                
                # Track significant drivers
                if abs(indicator_impact) > 0.5:
                    direction = '↑' if roc > 0 else '↓'
                    drivers.append({
                        'indicator': indicator_id,
                        'direction': direction,
                        'impact': round(indicator_impact, 2),
                        'sensitivity': sensitivity
                    })
    
    # Normalize impact score
    if total_weight > 0:
        normalized_impact = impact_score / total_weight
    else:
        normalized_impact = 0
    
    # Apply regime adjustment
    regime_adjustment = 0
    if sector_id in macro_regime.get('favored_sectors', []):
        regime_adjustment = 1.5
        drivers.append({
            'indicator': 'REGIME',
            'direction': '✓',
            'impact': 1.5,
            'note': f"Favored in {macro_regime['regime'].replace('_', ' ')}"
        })
    elif sector_id in macro_regime.get('avoid_sectors', []):
        regime_adjustment = -1.5
        drivers.append({
            'indicator': 'REGIME',
            'direction': '✗',
            'impact': -1.5,
            'note': f"Unfavored in {macro_regime['regime'].replace('_', ' ')}"
        })
    
    final_score = normalized_impact + regime_adjustment
    
    # Convert score to signal
    if final_score > 1:
        signal = 'BULLISH'
        signal_emoji = '🟢'
    elif final_score > 0.3:
        signal = 'LEAN_BULLISH'
        signal_emoji = '🟡'
    elif final_score > -0.3:
        signal = 'NEUTRAL'
        signal_emoji = '⚪'
    elif final_score > -1:
        signal = 'LEAN_BEARISH'
        signal_emoji = '🟠'
    else:
        signal = 'BEARISH'
        signal_emoji = '🔴'
    
    # Calculate confidence based on data availability and agreement
    driver_agreement = len([d for d in drivers if d['impact'] > 0]) / max(len(drivers), 1)
    data_coverage = total_weight / 5  # Normalized by expected indicators
    confidence = min((driver_agreement * 50 + data_coverage * 50), 95)
    
    return {
        'sector': sector_id,
        'name': sector_info['name'],
        'icon': sector_info['icon'],
        'signal': signal,
        'signal_emoji': signal_emoji,
        'score': round(final_score, 2),
        'confidence': round(confidence, 1),
        'horizon': f"{horizon_months} months",
        'drivers': sorted(drivers, key=lambda x: abs(x['impact']), reverse=True)[:5],
        'description': sensitivities.get('description', '')
    }


def predict_all_sectors(
    indicators_data: Dict,
    horizon_months: int = 9
) -> Dict:
    """
    Generate predictions for all sectors
    
    Args:
        indicators_data: Dict of all indicator derivatives
        horizon_months: Prediction horizon (6-12)
    
    Returns:
        Complete sector impact predictions
    """
    # First determine macro regime
    macro_regime = calculate_macro_regime(indicators_data)
    
    # Calculate impact for each sector
    sector_predictions = {}
    for sector_id in SECTOR_ETFS.keys():
        sector_predictions[sector_id] = calculate_sector_impact(
            sector_id,
            indicators_data,
            macro_regime,
            horizon_months
        )
    
    # Sort by score
    sorted_sectors = sorted(
        sector_predictions.values(),
        key=lambda x: x.get('score', 0),
        reverse=True
    )
    
    return {
        'regime': macro_regime,
        'predictions': sector_predictions,
        'ranked': sorted_sectors,
        'horizon_months': horizon_months,
        'top_picks': [s for s in sorted_sectors if s.get('signal') in ['BULLISH', 'LEAN_BULLISH']][:3],
        'avoid': [s for s in sorted_sectors if s.get('signal') in ['BEARISH', 'LEAN_BEARISH']][:3]
    }


def generate_summary_report(predictions: Dict) -> str:
    """Generate human-readable summary of sector predictions"""
    regime = predictions['regime']
    
    report = f"""
## Macro Regime: {regime['regime'].replace('_', ' ').title()}
**Confidence:** {regime['confidence']}%
**Description:** {regime['description']}

### Top Sector Picks (6-12 Month Outlook)
"""
    
    for sector in predictions['top_picks']:
        report += f"- {sector['signal_emoji']} **{sector['name']}** ({sector['sector']}): {sector['signal']} | Confidence: {sector['confidence']}%\n"
    
    report += "\n### Sectors to Avoid\n"
    
    for sector in predictions['avoid']:
        report += f"- {sector['signal_emoji']} **{sector['name']}** ({sector['sector']}): {sector['signal']} | Confidence: {sector['confidence']}%\n"
    
    return report
