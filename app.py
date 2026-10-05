import streamlit as st
import os, whisper, time, threading
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from moviepy.editor import VideoFileClip, concatenate_videoclips
if 'DB' not in globals(): globals()['DB'] = {}
g_db = globals()['DB']
@st.cache_resource
def load_whisper_model(): return whisper.load_model('base')
def ai_worker(task_id, in_path, out_path):
    try:
        g_db[task_id]['status'] = 'Processing: Cutting...'
        video = VideoFileClip(in_path)
        audio = AudioSegment.from_file(in_path)
        chunks = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-45)
        clips = [video.subclip(max(0, s/1000.0 - 0.1), min(video.duration, e/1000.0 + 0.1)) for s, e in chunks]
        if not clips: g_db[task_id]['status'] = 'Failed: No Voice'; video.close(); return
        final = concatenate_videoclips(clips)
        final.write_videofile(out_path, codec='libx264', audio_codec='aac')
        video.close(); final.close()
        g_db[task_id]['status'] = 'Processing: AI Subtitling...'
        model = load_whisper_model()
        res = model.transcribe(out_path, language='ko')
        txt = ''
        for s in res.get('segments', []):
            txt += f"[{s.get('start', 0.0):.2f}s] {s.get('text', '')}\n"
        g_db[task_id]['status'] = 'Success'; g_db[task_id]['result'] = txt
    except Exception as e: g_db[task_id]['status'] = f'Error: {str(e)}'
    finally:
        time.sleep(1.0)
        if os.path.exists(in_path): 
            try: os.remove(in_path)
            except: pass
st.set_page_config(layout='wide')
st.title('🚀 EasyCut AI Platform - Operator Mode')
col1, col2 = st.columns(2)
with col1:
    st.header('📥 Upload Video')
    up = st.file_uploader('Upload video', type=['mp4', 'mov'])
    if up and st.button('🚀 Start AI Edit', type='primary'):
        task_id = up.name
        if task_id not in g_db:
            in_p, out_p = f'storage_in_{task_id}', f'storage_out_{task_id}'
            with open(in_p, 'wb') as f: f.write(up.read())
            g_db[task_id] = {'status': 'Waiting', 'result': None, 'out_path': out_p}
            threading.Thread(target=ai_worker, args=(task_id, in_p, out_p)).start()
            st.rerun()
with col2:
    st.header('📊 Real-time Server Queue Status')
    if not g_db: st.write('No active tasks.')
    for tid, info in list(g_db.items()):
        with st.expander(f'📋 {tid} [ {info["status"]} ]', expanded=True):
            if info['status'] == 'Success':
                st.video(info['out_path'])
                st.text_area('AI Script Result', info['result'], height=150)
            elif 'Processing' in info['status'] or 'Waiting' in info['status']:
                time.sleep(3); st.rerun()
            elif 'Error' in info['status'] or 'Failed' in info['status']: st.error(info['status'])
