from __future__ import annotations

from datetime import datetime
import html
import json


def make_html_report(result: dict, language: str = "ro") -> str:
    ro = language.lower().startswith("ro")

    def tr(ro_text: str, en_text: str) -> str:
        return ro_text if ro else en_text

    def val(v, nd=2):
        if v is None:
            return tr("indisponibil", "not available")
        if isinstance(v, float):
            return f"{v:,.{nd}f}"
        return str(v)

    rows = []
    for section, items in result.items():
        if not isinstance(items, dict):
            continue
        rows.append(f"<h2>{html.escape(section.replace('_',' ').title())}</h2><table>")
        for k, v in items.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            rows.append(f"<tr><th>{html.escape(k.replace('_',' ').title())}</th><td>{html.escape(val(v))}</td></tr>")
        rows.append("</table>")

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = tr("Portalul incendiilor de pădure din România – Raport AOI", "Romania Forest Wildfire Portal – AOI report")
    return f"""<!doctype html>
<html lang='{language}'><head><meta charset='utf-8'><title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;max-width:960px;margin:40px auto;color:#18221c;line-height:1.45}}
h1{{color:#9b2f1f}} h2{{margin-top:26px;border-bottom:2px solid #eee;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}} th{{width:38%;background:#f6f7f5}}
.note{{background:#fff4e8;border-left:4px solid #dd7a22;padding:12px;margin:18px 0}}
.source{{background:#f4f7f5;padding:12px;border-radius:8px}}
.project{{background:#f7ece8;border-left:4px solid #9b2f1f;padding:12px;margin:18px 0}}
</style></head><body>
<h1>🔥 {title}</h1>
<div class='project'><b>PREPARE · WP4 – {tr('Incendii de pădure', 'Forest fires')}</b><br>
{tr('Prototip de suport pentru modelarea, monitorizarea, prevenția și evaluarea efectelor asociate incendiilor de pădure.', 'Decision-support prototype for modelling, monitoring, prevention and assessment of wildfire-related effects.')}<br>
<a href='https://utcb.ro/cercetare/programe-uefiscdi/pn-iv-pro-coex-2024-1-prepare/'>PREPARE</a></div>
<p>{tr('Generat', 'Generated')}: {generated}</p>
<div class='note'><b>{tr('Interpretare', 'Interpretation')}:</b> {tr('Scorul de screening al portalului este un indice transparent de suport decizional, nu Fire Weather Index oficial. Pentru pericolul oficial armonizat la nivel european utilizați stratul Copernicus EFFIS FWI.', 'The portal screening score is a transparent decision-support index, not the official Fire Weather Index. Use the Copernicus EFFIS FWI layer for the official harmonized European fire-danger product.')}</div>
{''.join(rows)}
<h2>{tr('Proveniența datelor', 'Data provenance')}</h2>
<div class='source'>
<p><b>NASA FIRMS:</b> {tr('anomalii termice VIIRS/MODIS aproape în timp real și Fire Radiative Power.', 'near-real-time VIIRS/MODIS thermal anomalies and Fire Radiative Power.')}</p>
<p><b>Copernicus EFFIS:</b> {tr('Fire Weather Index european și informații despre suprafețele arse.', 'official European Fire Weather Index and burned-area information.')}</p>
<p><b>Open-Meteo:</b> {tr('date meteorologice modelate, VPD și umiditatea solului.', 'modelled weather, vapour pressure deficit and soil moisture.')}</p>
<p><b>ESA CCI Biomass:</b> {tr('biomasă supraterană v7.0 pentru 2024, 100 m, accesată din arhiva publică CEDA.', 'v7.0 above-ground biomass for 2024 at 100 m, accessed from the public CEDA archive.')}</p>
<p><b>ESA WorldCover:</b> {tr('WorldCover 2021 v200, 10 m, analizat din COG-uri publice și afișat prin serviciul Terrascope.', 'WorldCover 2021 v200 at 10 m, analysed from public COGs and displayed through Terrascope.')}</p>
</div>
<p><b>{tr('Important', 'Important')}:</b> {tr('Produsele de teledetecție au incertitudini și nu trebuie substituite inventarului forestier sau informațiilor operative ale autorităților.', 'Remote-sensing products contain uncertainty and should not substitute a field forest inventory or operational information from competent authorities.')}</p>
</body></html>"""
