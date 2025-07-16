from PIL import Image
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components

def show_question_images(doc, pages):
    cols = st.columns(len(pages))
    for i, p in enumerate(pages):
        page = doc[p]
        pix = page.get_pixmap(dpi=250)
        image = Image.open(BytesIO(pix.tobytes("png")))
        cols[i].image(image, caption=f"페이지 {p}", use_container_width=True)

def render_js_timer(timer_id):
    components.html(f"""
        <div id="timer_{timer_id}" style="font-size:24px; font-weight:bold; color:green; margin: 10px 0;"></div>
        <script>
        if (!sessionStorage.getItem('remaining_{timer_id}')) {{
            sessionStorage.setItem('remaining_{timer_id}', 0);
        }}
        var total = parseInt(sessionStorage.getItem('remaining_{timer_id}'));
        if (isNaN(total)) total = 0;

        function updateTimer() {{
            var paused = sessionStorage.getItem('paused_{timer_id}') === 'true';
            var minutes = Math.floor(total / 60);
            var seconds = total % 60;
            if (paused) {{
                document.getElementById("timer_{timer_id}").innerHTML = "⏸ 타이머가 일시 정지되었습니다. 남은 시간: " + minutes + "분 " + (seconds < 10 ? "0" : "") + seconds + "초";
            }} else if (total > 0) {{
                document.getElementById("timer_{timer_id}").innerHTML = "남은 시간: " + minutes + "분 " + (seconds < 10 ? "0" : "") + seconds + "초";
                total -= 1;
                sessionStorage.setItem('remaining_{timer_id}', total);
            }} else {{
                document.getElementById("timer_{timer_id}").innerHTML = "⏰ 시간이 종료되었습니다!";
            }}
            setTimeout(updateTimer, 1000);
        }}
        updateTimer();
        </script>
    """, height=60)
