"""
DAPS Certification Lab — Generador de reportes.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Genera reportes HTML y JSON."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config['reports']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        backtest_result,
        metrics: Dict,
        tier_metrics: pd.DataFrame,
        hourly_metrics: pd.DataFrame,
        symbol_metrics: pd.DataFrame,
        mc_results: Dict,
        wf_results: Dict,
        risk_metrics: Dict,
        optimization_results: pd.DataFrame = None,
    ) -> Dict[str, str]:
        """Genera reporte completo en todos los formatos."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'metrics': metrics,
            'risk_metrics': risk_metrics,
            'tier_metrics': tier_metrics.to_dict('records') if not tier_metrics.empty else [],
            'hourly_metrics': hourly_metrics.to_dict('records') if not hourly_metrics.empty else [],
            'symbol_metrics': symbol_metrics.to_dict('records') if not symbol_metrics.empty else [],
            'monte_carlo': mc_results,
            'walk_forward': wf_results,
        }

        outputs = {}

        # JSON
        json_path = self.output_dir / f'daps_report_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        outputs['json'] = str(json_path)

        # HTML
        html_path = self.output_dir / f'daps_report_{timestamp}.html'
        html = self._render_html(report)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        outputs['html'] = str(html_path)

        # Trades CSV
        if backtest_result and backtest_result.trades:
            trades_df = pd.DataFrame([t.__dict__ for t in backtest_result.trades])
            csv_path = self.output_dir / f'daps_trades_{timestamp}.csv'
            trades_df.to_csv(csv_path, index=False)
            outputs['trades_csv'] = str(csv_path)

        # Optimization CSV
        if optimization_results is not None and not optimization_results.empty:
            opt_path = self.output_dir / f'daps_optimization_{timestamp}.csv'
            optimization_results.to_csv(opt_path, index=False)
            outputs['optimization_csv'] = str(opt_path)

        logger.info(f"✅ Reportes generados en {self.output_dir}")
        return outputs

    def _render_html(self, report: Dict) -> str:
        """Renderiza reporte HTML."""
        m = report.get('metrics', {})
        mc = report.get('monte_carlo', {})
        wf = report.get('walk_forward', {})

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>DAPS Omega Certification Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; border-left: 4px solid #4361ee; padding-left: 10px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #4361ee; color: white; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.metric {{ display: inline-block; background: #f0f4ff; padding: 15px 25px; margin: 5px; border-radius: 8px; }}
.metric-value {{ font-size: 24px; font-weight: bold; color: #4361ee; }}
.metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
.certified {{ background: #d4edda; color: #155724; padding: 15px; border-radius: 8px; margin: 20px 0; }}
.rejected {{ background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin: 20px 0; }}
</style></head><body>
<h1>🔬 DAPS Omega — Certification Report</h1>
<p><strong>Fecha:</strong> {report.get('timestamp', '')}</p>
<p><strong>Versión:</strong> {report.get('config', {}).get('project', {}).get('version', 'N/A')}</p>

<h2>📊 Métricas Globales</h2>
<div>
<div class="metric"><div class="metric-value">{m.get('total_trades', 0)}</div><div class="metric-label">Trades</div></div>
<div class="metric"><div class="metric-value">{m.get('win_rate', 0):.1f}%</div><div class="metric-label">Win Rate</div></div>
<div class="metric"><div class="metric-value">{m.get('profit_factor', 0):.2f}</div><div class="metric-label">Profit Factor</div></div>
<div class="metric"><div class="metric-value">{m.get('sharpe', 0):.2f}</div><div class="metric-label">Sharpe</div></div>
<div class="metric"><div class="metric-value">{m.get('max_drawdown_pct', 0):.2f}%</div><div class="metric-label">Max DD</div></div>
<div class="metric"><div class="metric-value">${m.get('final_equity', 0):,.0f}</div><div class="metric-label">Final Equity</div></div>
</div>

<h2>🎲 Monte Carlo ({mc.get('n_iterations', 0)} iteraciones)</h2>
<table>
<tr><th>Métrica</th><th>Valor</th></tr>
<tr><td>Probabilidad positiva</td><td>{mc.get('prob_positive', 0) * 100:.1f}%</td></tr>
<tr><td>Retorno medio</td><td>{mc.get('final_return_mean', 0) * 100:.2f}%</td></tr>
<tr><td>Retorno CI95%</td><td>{mc.get('final_return_ci95', (0, 0))}</td></tr>
<tr><td>Max DD medio</td><td>{mc.get('max_dd_mean', 0) * 100:.2f}%</td></tr>
<tr><td>Prob DD > 20%</td><td>{mc.get('prob_dd_gt_20', 0) * 100:.1f}%</td></tr>
</table>

<h2>🔄 Walk Forward</h2>
<table>
<tr><th>Métrica</th><th>Valor</th></tr>
<tr><td>Ventanas</td><td>{wf.get('n_windows', 0)}</td></tr>
<tr><td>Positivas</td><td>{wf.get('positive_windows', 0)} ({wf.get('positive_pct', 0):.1f}%)</td></tr>
<tr><td>Retorno medio</td><td>{wf.get('mean_return_pct', 0):.2f}%</td></tr>
<tr><td>Sharpe medio</td><td>{wf.get('mean_sharpe', 0):.2f}</td></tr>
</table>

<h2>🏷️ Por Tier</h2>
<table>
<tr><th>Tier</th><th>Trades</th><th>WR</th><th>PF</th><th>Expectancy</th></tr>
"""
        for row in report.get('tier_metrics', []):
            html += f"<tr><td>{row.get('tier', '')}</td><td>{row.get('trades', 0)}</td>" \
                    f"<td>{row.get('win_rate', 0):.1f}%</td>" \
                    f"<td>{row.get('profit_factor', 0):.2f}</td>" \
                    f"<td>{row.get('expectancy', 0):.3f}%</td></tr>"

        html += """
</table>
</body></html>"""
        return html
