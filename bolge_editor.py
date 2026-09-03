"""
============================================================
POTA TAKİP — İşlem Bölgesi Çizim Arayüzü
============================================================
Her kameranın karesini tarayıcıda gösterir; fareyle İŞLEM BÖLGELERİNİ
(dikdörtgen — kare de olur, ray gibi uzun şerit de) çizip kaydedersin.
Sistem sonra sadece bu bölgelerdeki potaları sayar.

KURULUM:  pip install flask
ÇALIŞTIR: python bolge_editor.py
          sonra tarayıcıda:  http://localhost:8001

Kaydedilen bölgeler veri/<video>_bolgeler.json dosyasına yazılır ve
'bolge_tarama.py analiz/isle' bunları otomatik kullanır (otomatik öneri
yerine senin çizdiklerin geçerli olur).
============================================================
"""

import os
import glob
import json

from flask import Flask, request, jsonify, send_file, render_template_string

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VERI_DIR = os.path.join(PROJECT_DIR, "veri")

app = Flask(__name__)


def cameras():
    """veri/ içindeki *_frame.jpg dosyalarından kamera listesi çıkarır."""
    out = []
    for fp in sorted(glob.glob(os.path.join(VERI_DIR, "*_frame.jpg"))):
        base = os.path.basename(fp)[:-len("_frame.jpg")]
        # kamera adını okunur hale getir
        ad = base.replace("NVR_", "").replace("", "")
        out.append({"base": base, "ad": ad})
    return out


@app.route("/")
def index():
    return render_template_string(PAGE, cameras=cameras())


@app.route("/kare/<base>")
def kare(base):
    fp = os.path.join(VERI_DIR, base + "_frame.jpg")
    if not os.path.exists(fp):
        return "kare yok", 404
    return send_file(fp, mimetype="image/jpeg")


@app.route("/isi/<base>")
def isi(base):
    """Isı haritası (potaların nerede yoğunlaştığı) — varsa. Yoksa düz kareyi döndür."""
    fp = os.path.join(VERI_DIR, base + "_isihalitasi.jpg")
    if not os.path.exists(fp):
        fp = os.path.join(VERI_DIR, base + "_frame.jpg")
    if not os.path.exists(fp):
        return "yok", 404
    return send_file(fp, mimetype="image/jpeg")


@app.route("/isi-var/<base>")
def isi_var(base):
    """Bu kamera taranmış mı (ısı haritası var mı)?"""
    fp = os.path.join(VERI_DIR, base + "_isihalitasi.jpg")
    return jsonify({"var": os.path.exists(fp)})


@app.route("/bolgeler/<base>")
def bolgeler_get(base):
    """Bu kamera için kaydedilmiş bölgeleri döndür (varsa)."""
    fp = os.path.join(VERI_DIR, base + "_bolgeler.json")
    if not os.path.exists(fp):
        return jsonify({"zones": [], "tipler": []})
    data = json.load(open(fp))
    zones = data.get("zones", [])
    # eski dosyalarda "tipler" yok — hepsi "islem" (varsayılan, pota_canli_takip.py'nin
    # kullandığı işlem-bölgesi anlamı) kabul edilir, geriye dönük uyumluluk için
    tipler = data.get("tipler") or ["islem"] * len(zones)
    return jsonify({"zones": zones, "tipler": tipler})


@app.route("/kaydet/<base>", methods=["POST"])
def kaydet(base):
    """Çizilen bölgeleri veri/<base>_bolgeler.json'a yaz.

    "tipler": her bölgenin ne anlama geldiği (zones ile aynı sırada/uzunlukta):
      "islem"    → pota_canli_takip.py'nin kullandığı işlem/doluluk bölgesi (varsayılan)
      "kapi_sol" / "kapi_sag" → v2'nin giriş/çıkış kapı doğrulaması için
      "gecersiz" → bu bölgedeki tespitler gürültü sayılır, tamamen göz ardı edilir
    "tipler" gönderilmezse (eski istemci) hepsi "islem" kabul edilir — mevcut
    pota_canli_takip.py davranışı hiç değişmez.
    """
    body = request.get_json(force=True)
    zones = body.get("zones", [])
    tipler = body.get("tipler") or ["islem"] * len(zones)
    # boyutu frame'den değil, tespitler dosyasından al (asıl çözünürlük)
    width = body.get("width", 2560)
    height = body.get("height", 1440)
    fp = os.path.join(VERI_DIR, base + "_bolgeler.json")
    with open(fp, "w") as f:
        json.dump({"zones": zones, "tipler": tipler, "width": width, "height": height,
                   "kaynak": "elle_cizildi"}, f, indent=2)
    return jsonify({"ok": True, "adet": len(zones), "dosya": os.path.basename(fp)})


PAGE = r"""
<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>İşlem Bölgesi Çizimi</title>
<style>
  :root{--bg:#0e1113;--card:#171b1e;--fg:#e8ecea;--muted:#98a5a0;--border:#2a3033;
        --accent:#2f9c67;--danger:#f87171;}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--fg);font-family:-apple-system,system-ui,sans-serif;font-size:14px}
  .wrap{max-width:1200px;margin:0 auto;padding:16px}
  header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px}
  h1{font-size:17px;font-weight:600}
  select{background:var(--card);color:var(--fg);border:1px solid var(--border);
         border-radius:6px;padding:7px 10px;font-size:14px;max-width:420px}
  .toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  button{background:var(--card);color:var(--fg);border:1px solid var(--border);
         border-radius:6px;padding:7px 12px;font-size:14px;cursor:pointer;transition:.12s}
  button:hover{border-color:var(--muted)}
  button.primary{background:var(--accent);border-color:var(--accent);color:#06120c;font-weight:600}
  button.danger{color:var(--danger);border-color:#48292b}
  .hint{color:var(--muted);font-size:13px;line-height:1.6;margin-bottom:10px}
  .canvas-wrap{position:relative;display:inline-block;border:1px solid var(--border);border-radius:8px;overflow:hidden;max-width:100%}
  canvas{display:block;max-width:100%;cursor:crosshair}
  .msg{margin-top:10px;font-size:13px;color:var(--accent);min-height:18px}
  .zonelist{margin-top:12px;display:flex;flex-direction:column;gap:6px}
  .zone-item{display:flex;align-items:center;gap:10px;background:var(--card);border:1px solid var(--border);
             border-radius:6px;padding:6px 10px;font-size:13px}
  .zone-item .sw{width:14px;height:14px;border-radius:3px}
  .zone-item .rm{margin-left:auto;color:var(--danger);cursor:pointer;font-size:12px}
</style></head><body>
<div class="wrap">
  <header>
    <h1>İşlem Bölgesi Çizimi</h1>
    <select id="cam" onchange="camChange()">
      {% for c in cameras %}<option value="{{c.base}}">{{c.ad}}</option>{% endfor %}
    </select>
  </header>

  <div class="hint">
    Aşağıdan çizeceğin bölgenin <b>tipini</b> seç, sonra fareyle sürükleyerek çiz
    (kare veya ray gibi uzun şerit). Birden fazla bölge çizebilirsin. Bitince
    <b>Kaydet</b>'e bas.
  </div>

  <div class="toolbar">
    <select id="tipSec">
      <option value="islem">İşlem Bölgesi</option>
      <option value="kapi_sol">Kapı (Sol)</option>
      <option value="kapi_sag">Kapı (Sağ)</option>
      <option value="gecersiz">Geçersiz Bölge (gürültü)</option>
    </select>
    <button class="primary" onclick="kaydet()">💾 Kaydet</button>
    <button id="isiBtn" onclick="isiToggle()">🔥 Isı haritasını göster</button>
    <button class="danger" onclick="temizle()">Tümünü sil</button>
    <button onclick="sonuncuyuSil()">Son çizimi geri al</button>
  </div>
  <div class="hint">
    <b>İşlem Bölgesi</b>: pota_canli_takip.py'nin doluluk taradığı alan.
    <b>Kapı (Sol/Sağ)</b>: v2'nin giriş=çıkış yönü doğrulaması için kadrajın
    giriş/çıkış raylarını işaretle. <b>Geçersiz Bölge</b>: yansıma/kıvılcım gibi
    bilinen gürültü kaynaklarını işaretle — buradaki tespitler tamamen göz ardı edilir.
  </div>
  <div class="hint" id="isiHint" style="display:none">
    🔥 Sıcak (kırmızı/sarı) bölgeler potaların en çok göründüğü yerler.
    <b>Bölgeni bu sıcak alanların üzerine çiz</b> — potanın merkezi oraya düşmeli.
  </div>

  <div class="canvas-wrap"><canvas id="cv"></canvas></div>
  <div class="msg" id="msg"></div>
  <div class="zonelist" id="zonelist"></div>
</div>

<script>
const TIP_RENK = {islem:"#2f9c67", kapi_sol:"#00b4ff", kapi_sag:"#ff7828", gecersiz:"#f87171"};
const TIP_AD = {islem:"İşlem Bölgesi", kapi_sol:"Kapı (Sol)", kapi_sag:"Kapı (Sağ)", gecersiz:"Geçersiz Bölge"};
let img = new Image();
let zones = [];         // her biri [x1,y1,x2,y2] ORİJİNAL çözünürlükte
let tipler = [];        // zones ile aynı sırada, her birinin tipi ("islem"/"kapi_sol"/...)
let natW=2560, natH=1440;
let drawing=false, sx=0, sy=0, cx=0, cy=0;
let isiAcik=false;
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");

function camBase(){ return document.getElementById("cam").value; }

function camChange(){
  const base = camBase();
  img = new Image();
  img.onload = ()=>{
    natW = img.naturalWidth; natH = img.naturalHeight;
    // canvası ekrana sığdır (en fazla 1100 px genişlik)
    const dispW = Math.min(1100, natW);
    cv.width = dispW; cv.height = Math.round(natH * dispW/natW);
    loadZones(base);
  };
  img.src = (isiAcik ? "/isi/" : "/kare/") + encodeURIComponent(base);
}

function isiToggle(){
  isiAcik = !isiAcik;
  document.getElementById("isiBtn").textContent =
    isiAcik ? "🔥 Isı haritasını gizle" : "🔥 Isı haritasını göster";
  document.getElementById("isiHint").style.display = isiAcik ? "block" : "none";
  // arka planı değiştir ama çizimleri koru
  const base = camBase();
  const yeni = new Image();
  yeni.onload = ()=>{ img = yeni; redraw(); };
  yeni.src = (isiAcik ? "/isi/" : "/kare/") + encodeURIComponent(base) + "?t=" + Date.now();
}

function loadZones(base){
  fetch("/bolgeler/"+encodeURIComponent(base)).then(r=>r.json()).then(d=>{
    zones = d.zones || [];
    tipler = d.tipler || zones.map(()=>"islem");
    redraw();
  });
}

function sc(){ return cv.width / natW; }   // ekran/orijinal oranı

function redraw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  const s = sc();
  zones.forEach((z,i)=>{
    const c = TIP_RENK[tipler[i]] || TIP_RENK.islem;
    ctx.strokeStyle=c; ctx.lineWidth=3;
    ctx.strokeRect(z[0]*s, z[1]*s, (z[2]-z[0])*s, (z[3]-z[1])*s);
    ctx.fillStyle=c; ctx.font="bold 15px system-ui";
    ctx.fillText(TIP_AD[tipler[i]] || "Bölge "+(i+1), z[0]*s+4, z[1]*s-6<12? z[1]*s+16 : z[1]*s-6);
  });
  if(drawing){
    ctx.strokeStyle="#ffffff"; ctx.setLineDash([6,4]); ctx.lineWidth=2;
    ctx.strokeRect(sx, sy, cx-sx, cy-sy); ctx.setLineDash([]);
  }
  renderList();
}

function renderList(){
  const el = document.getElementById("zonelist");
  el.innerHTML = zones.map((z,i)=>{
    const c = TIP_RENK[tipler[i]] || TIP_RENK.islem;
    const w = Math.abs(z[2]-z[0]), h = Math.abs(z[3]-z[1]);
    const sekil = w > h*2.2 ? "yatay şerit (ray)" : (h>w*2.2? "dikey şerit":"kutu");
    const secenekler = Object.keys(TIP_AD).map(t=>
      `<option value="${t}" ${tipler[i]===t?"selected":""}>${TIP_AD[t]}</option>`).join("");
    return `<div class="zone-item"><span class="sw" style="background:${c}"></span>
      Bölge ${i+1} — ${w}×${h} px (${sekil})
      <select onchange="tipDegistir(${i}, this.value)">${secenekler}</select>
      <span class="rm" onclick="silBir(${i})">sil</span></div>`;
  }).join("");
}

function tipDegistir(i, yeniTip){ tipler[i] = yeniTip; redraw(); }

cv.addEventListener("mousedown", e=>{
  const r=cv.getBoundingClientRect();
  sx=e.clientX-r.left; sy=e.clientY-r.top; cx=sx; cy=sy; drawing=true;
});
cv.addEventListener("mousemove", e=>{
  if(!drawing) return;
  const r=cv.getBoundingClientRect();
  cx=e.clientX-r.left; cy=e.clientY-r.top; redraw();
});
cv.addEventListener("mouseup", e=>{
  if(!drawing) return;
  drawing=false;
  const s=sc();
  let x1=Math.min(sx,cx)/s, y1=Math.min(sy,cy)/s, x2=Math.max(sx,cx)/s, y2=Math.max(sy,cy)/s;
  if(Math.abs(x2-x1)>10 && Math.abs(y2-y1)>10){
    zones.push([Math.round(x1),Math.round(y1),Math.round(x2),Math.round(y2)]);
    tipler.push(document.getElementById("tipSec").value);
  }
  redraw();
});

function silBir(i){ zones.splice(i,1); tipler.splice(i,1); redraw(); }
function sonuncuyuSil(){ zones.pop(); tipler.pop(); redraw(); }
function temizle(){ zones=[]; tipler=[]; redraw(); }

function kaydet(){
  fetch("/kaydet/"+encodeURIComponent(camBase()), {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({zones:zones, tipler:tipler, width:natW, height:natH})
  }).then(r=>r.json()).then(d=>{
    document.getElementById("msg").textContent =
      `✔ ${d.adet} bölge kaydedildi (${d.dosya}). Artık 'analiz' bunları kullanacak.`;
  });
}

camChange();  // ilk kamerayı yükle
</script>
</body></html>
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  İşlem Bölgesi Çizim Arayüzü")
    print("  Tarayıcıda aç:  http://localhost:8001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8001, debug=False)
