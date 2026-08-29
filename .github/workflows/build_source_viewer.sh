#!/usr/bin/env bash
set -euo pipefail
npm install -g cordova
rm -rf app
cordova create app com.noor.universal "noor"
cd app
cordova platform add android
rm -rf www/*
mkdir -p www
python3 - <<'PY'
from pathlib import Path
import html,json
root=Path('../project')
allowed={'.tsx','.ts','.jsx','.js','.py','.php','.cpp','.cc','.cxx','.c','.h','.hpp','.properties','.md','.kts','.xml','.kt','.java','.gradle','.dart','.html','.css','.scss','.json','.yaml','.yml','.toml','.ini','.txt','.sh','.bat'}
items=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or any(x in {'.git','.gradle','build','node_modules','__MACOSX'} for x in p.parts): continue
    if p.suffix.lower() in allowed or p.name.lower() in {'gradlew','gradle.properties','settings.gradle','settings.gradle.kts','build.gradle','build.gradle.kts','pubspec.yaml','package.json'}:
        try: t=p.read_text(encoding='utf-8',errors='replace') if p.stat().st_size<=400000 else '[File too large for preview]'
        except Exception as e: t=f'[Read error: {e}]'
        items.append({'path':p.relative_to(root).as_posix(),'content':t})
data=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
name=html.escape('noor')
page=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name}</title><style>body{{margin:0;background:#101114;color:#eee;font-family:system-ui}}header{{padding:15px;font-weight:700;background:#17181c;position:sticky;top:0}}input{{margin:10px;padding:12px;width:calc(100% - 44px);border-radius:10px;border:1px solid #444;background:#202228;color:#fff}}button{{display:block;width:100%;text-align:left;padding:12px;border:0;border-bottom:1px solid #292b31;background:#15161a;color:#ddd}}pre{{white-space:pre-wrap;word-break:break-word;padding:14px;font-size:12px;line-height:1.5}}.path{{color:#8ab4f8}}</style></head><body><header>🧩 {name} — Universal Source APK</header><input id="q" placeholder="Search files..."><div id="list"></div><pre id="code">Select a file</pre><script>const files={data},list=document.getElementById('list'),code=document.getElementById('code'),q=document.getElementById('q');function esc(s){{return s.replace(/&/g,'&amp;').replace(/</g,'&lt;')}}function render(){{list.innerHTML='';files.filter(x=>x.path.toLowerCase().includes(q.value.toLowerCase())).forEach(x=>{{let b=document.createElement('button');b.textContent=x.path;b.onclick=()=>code.innerHTML='<span class="path">'+esc(x.path)+'</span>\\n\\n'+esc(x.content);list.appendChild(b)}})}}q.oninput=render;render();</script></body></html>'''
Path('www/index.html').write_text(page,encoding='utf-8')
PY
cordova build android --debug
mkdir -p ../out
cp platforms/android/app/build/outputs/apk/debug/*.apk ../out/app.apk
