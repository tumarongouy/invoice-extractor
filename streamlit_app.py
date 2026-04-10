import streamlit as st
from google import genai
import pandas as pd
import json
import requests
from PIL import Image
import io
import re
import os
from dotenv import load_dotenv

# Load local .env
load_dotenv()

# --- UI Setup ---
st.set_page_config(page_title="Invoice to Excel AI", layout="wide")
st.title("🧾 Invoice Extractor (Production Ready)")

# --- Secret Management ---
# ดึง Key จาก ENV หรือ Streamlit Secrets
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")

# --- Sidebar Config ---
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("เลือก AI Provider", ["Gemini (Google)", "OpenRouter (Gemini 2.0)"])
    
    # กำหนด API Key อัตโนมัติถ้ามีในระบบ
    default_key = GEMINI_KEY if provider == "Gemini (Google)" else OPENROUTER_KEY
    
    api_key_input = st.text_input(
        "กรอก API Key (ปล่อยว่างหากใช้ Key ของระบบ)", 
        value=default_key if default_key else "",
        type="password"
    )
    
    # ใช้คีย์จาก Input ถ้ากรอกมา ถ้าไม่กรอกให้ใช้จากระบบ
    active_key = api_key_input if api_key_input else default_key
    
    if active_key:
        st.success(f"✅ พร้อมใช้งาน ({provider})")
    else:
        st.warning(f"⚠️ กรุณาตั้งค่า API Key สำหรับ {provider}")
    
    st.info("💡 แผนสำรอง: หาก Gemini ติด Error 429/503 ให้สลับไปใช้ OpenRouter แทน")

# --- Logic: AI Extraction ---
def extract_with_gemini(uploaded_file, key):
    client = genai.Client(api_key=key)
    
    # Read image bytes
    image_bytes = uploaded_file.getvalue()
    
    prompt = """คุณคือผู้เชี่ยวชาญด้านการสกัดข้อมูลใบแจ้งหนี้ สกัดข้อมูลตามกฎเหล็กดังนี้:
    1. ต้องสกัดข้อมูลมาทุกบรรทัด (Every single item row) ห้ามข้ามบรรทัดเลขที่ 1 และ 3 หรือบรรทัดใดๆ ทั้งสิ้นแม้ไม่มี S/N
    2. รูปแบบ JSON List: [{ "invoice_no": "", "date": "", "vendor": "", "grand_total": 0, "items": [{ "item_code": "รหัสสินค้าตัวแรกสุด", "desc": "ชื่อสินค้า", "qty": 0, "price": 0, "total": 0, "sn": ["S/N1", "S/N2"] }] }]
    3. หากพบ S/N ให้ใส่ใน "sn" เท่านั้น และห้ามเอาไปใส่ปนใน "desc"
    4. หากไม่มี S/N ให้ใส่ "sn": [] หรือ "" แต่ห้ามลบบรรทัดนั้นทิ้งเด็ดขาด
    5. สกัดชื่อ Vendor และ Invoice No ให้ถูกต้องจากหัวกระดาษ"""
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            prompt,
            genai.types.Part.from_bytes(data=image_bytes, mime_type=uploaded_file.type)
        ]
    )
    
    # Clean JSON output
    text = response.text
    # หา [ หรือ { ตัวแรก
    start_index = min(idx for idx in [text.find('['), text.find('{')] if idx != -1)
    if start_index == -1:
        raise Exception("AI did not return valid JSON: " + text)
    
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start_index:])
        # Ensure it's a list
        return data if isinstance(data, list) else [data]
    except Exception as e:
        raise Exception(f"JSON Parse Error: {str(e)}\nRaw Text: {text}")

def extract_with_openrouter(uploaded_file, key):
    import base64
    base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {key.strip()}",
        "HTTP-Referer": "http://localhost:8501", 
        "X-Title": "Invoice AI Extractor",
        "Content-Type": "application/json"
    }
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "google/gemini-2.0-flash-001",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract EVERY item line from the invoice into a JSON list. Rules: 1. Item Code is the first string (e.g., HWMAV...). 2. Extract ALL items including lines 1 and 3 (with or without S/N). 3. If multiple S/Ns exist for one item, put them in a list. 4. Format: [{invoice_no, date, vendor, grand_total, items: [{item_code, desc, qty, price, total, sn: []}]}]"},
                        {"type": "image_url", "image_url": {"url": f"data:{uploaded_file.type};base64,{base64_image}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }
    )
    result = response.json()
    if 'error' in result:
        raise Exception(result['error']['message'])
        
    content = result['choices'][0]['message']['content']
    if isinstance(content, str):
        start_index = min(idx for idx in [content.find('['), content.find('{')] if idx != -1)
        if start_index == -1:
            raise Exception("AI did not return valid JSON: " + content)
        try:
            data, _ = json.JSONDecoder().raw_decode(content[start_index:])
            return data if isinstance(data, list) else [data]
        except Exception as e:
            raise Exception(f"JSON Parse Error: {str(e)}\nRaw Content: {content}")
    return content if isinstance(content, list) else [content]

# --- Main App ---
uploaded_file = st.file_uploader("อัปโหลดภาพใบแจ้งหนี้ (JPG/PNG/PDF)", type=['jpg', 'jpeg', 'png', 'pdf'])

if uploaded_file and active_key:
    if st.button("🚀 เริ่มสกัดข้อมูล"):
        with st.spinner("AI กำลังอ่านข้อมูล..."):
            try:
                if provider == "Gemini (Google)":
                    data = extract_with_gemini(uploaded_file, active_key)
                else:
                    data = extract_with_openrouter(uploaded_file, active_key)
                
                # แสดงผลสรุป
                st.success(f"สกัดข้อมูลสำเร็จ! พบใบแจ้งหนี้ทั้งหมด {len(data)} ใบ")
                
                # รวมข้อมูลทั้งหมดเพื่อเข้าตารางเดียว
                all_rows = []
                for invoice in data:
                    inv_no = invoice.get('invoice_no', 'N/A')
                    date = invoice.get('date', 'N/A')
                    vendor = invoice.get('vendor', 'N/A')
                    grand = invoice.get('grand_total', 0)
                    
                    for item in invoice.get('items', []):
                        sn_data = item.get('sn', '')
                        # ถ้าเป็น list ของ S/N ให้แยกเป็นหลายแถว
                        sns = sn_data if isinstance(sn_data, list) else [sn_data]
                        
                        for single_sn in sns:
                            all_rows.append({
                                "Invoice No": inv_no,
                                "Date": date,
                                "Vendor": vendor,
                                "Item Code": item.get('item_code', ''),
                                "Description": item.get('desc', ''),
                                "S/N": single_sn,
                                "Qty": item.get('qty', 1) if len(sns) > 1 else item.get('qty', 0), # ถ้าแยกแถว Qty ควรเป็น 1
                                "Price": item.get('price', 0),
                                "Total": item.get('total', 0),
                                "Grand Total": grand
                            })
                
                df = pd.DataFrame(all_rows)
                
                st.subheader("Preview & Edit ข้อมูล (รวมจากทุกใบแจ้งหนี้)")
                edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True) 

                # ปุ่มโหลด Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    edited_df.to_excel(writer, index=False, sheet_name='All Invoices')
                
                # ตั้งชื่อไฟล์ตามเลขที่ใบแจ้งหนี้แรกที่พบ (ไม่มีคำว่า invoice_)
                first_invoice_no = data[0].get('invoice_no', 'export') if data else 'export'
                file_name_ready = f"{first_invoice_no}.xlsx"
                
                st.download_button(
                    label="📥 ดาวน์โหลดไฟล์ Excel (ทุกใบ)",
                    data=output.getvalue(),
                    file_name=file_name_ready,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
elif not active_key:
    st.warning("⚠️ กรุณาตั้งค่า API Key ในแถบด้านซ้ายก่อนเริ่มใช้งาน")
