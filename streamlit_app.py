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
    
    prompt = """คุณคือ AI สกัดข้อมูลใบแจ้งหนี้ระดับสูง (Invoice Specialist) 
    
    ### กฎเหล็ก (STRICT RULES):
    1. **REQUIRED ALL ITEMS**: ต้องสกัดข้อมูลมาให้ "ครบทุกบรรทัด" ในตาราง ห้ามข้ามเด็ดขาด
    2. **STRIKETHROUGH & MARKS**: หากพบตัวหนังสือที่มีเส้นขีดฆ่า (Strikethrough), เส้นสีแดงทับ หรือรอยปากกา "ให้สกัดข้อความนั้นออกมาเป็นข้อมูลจริง" ห้ามมองว่าเป็นข้อความที่ถูกยกเลิก
    3. **MISSING S/N**: รายการที่ไม่มี Serial Number (S/N) ให้สกัดออกมาด้วย โดยใส่ค่า sn เป็น array ว่าง `[]`
    4. **ITEM CODE CLARITY**: รหัสสินค้า (Item Code) ให้สังเกตจาก "ตัวอักษรภาษาอังกฤษตัวพิมพ์ใหญ่ตัวแรกสุด" ของแต่ละบรรทัด ตัวอย่างเช่น "HWMAV...", "SWMAV...", "SVC..." ฯลฯ หากมีข้อความอื่นบังหน้า ให้เริ่มสกัดตั้งแต่ตัวพิมพ์ใหญ่ตัวแรกเป็นต้นไป
    
    ### ตัวอย่างการสกัด (EXAMPLES):
    - กรณีเจอเส้นขีดฆ่า: ภาพมีคำว่า ~~HWMAV123~~ -> สกัดได้ "HWMAV123"
    - กรณีไม่มี S/N: รายการ MA Service (ไม่มี S/N) -> สกัดรายการตามปกติ, `sn: []`
    - กรณี Item Code: พบบรรทัด "1. HWMAV500" -> Item Code คือ "HWMAV500"

    ### รูปแบบ JSON ที่ต้องการ (RESPONSE FORMAT):
    ให้ส่งกลับมาเป็น List ของ Object เท่านั้น:
    [{
        "invoice_no": "เลขที่เอกสาร (เช่น IV-2024001)",
        "date": "วันที่ (เช่น 10/04/2024)",
        "vendor": "ชื่อบริษัทผู้ขาย",
        "grand_total": 0,
        "items": [
            { 
                "item_code": "HWMAV500012M", 
                "desc": "ชื่อสินค้าหรือรายละเอียด", 
                "qty": 1, 
                "price": 1000, 
                "total": 1000, 
                "sn": ["SN123", "SN456"] 
            }
        ]
    }]"""
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            prompt,
            genai.types.Part.from_bytes(data=image_bytes, mime_type=uploaded_file.type)
        ],
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    
    # Clean JSON output
    data = json.loads(response.text)
    return data if isinstance(data, list) else [data]

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
                        {
                            "type": "text", 
                            "text": """EXTRACT ALL ITEMS FROM INVOICE. 
                            RULES:
                            1. DO NOT SKIP ANY LINE even if it has red marks, strikethroughs, or no S/N.
                            2. ITEM CODE is the string starting with the first uppercase English letter in the line (e.g., 'HWMAV...').
                            3. If no S/N found, set 'sn' to [].
                            4. Treat strikethroughs or red lines as VALID DATA to be extracted.
                            Format as JSON: [{invoice_no, date, vendor, grand_total, items: [{item_code, desc, qty, price, total, sn: []}]}]"""
                        },
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
                    extracted_data = extract_with_gemini(uploaded_file, active_key)
                else:
                    extracted_data = extract_with_openrouter(uploaded_file, active_key)
                
                # เก็บข้อมูลลง session_state เพื่อให้ติ๊กกล่อง debug ได้โดยไม่หาย
                st.session_state['data'] = extracted_data
                st.success(f"สกัดข้อมูลสำเร็จ! พบข้อมูลทั้งหมด {len(extracted_data)} ชุด")
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")

# --- Display Results ---
if 'data' in st.session_state:
    data = st.session_state['data']
    
    # Checkbox สำหรับ Debug
    show_debug = st.checkbox("🔍 แสดง JSON ดิบ (สำหรับตรวจสอบ)")
    if show_debug:
        st.json(data)
    
    # รวมข้อมูลทั้งหมดเพื่อเข้าตารางเดียว
    all_rows = []
    for entry in data:
        # AI บางครั้งอาจส่งมาเป็น { "invoices": [...] } หรือ { "invoice": { ... } }
        invoices = []
        if isinstance(entry, dict):
            if "items" in entry:
                invoices = [entry]
            elif "invoices" in entry:
                invoices = entry["invoices"]
            elif "invoice" in entry:
                invoices = [entry["invoice"]]
            else:
                for val in entry.values():
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        invoices = [entry] 
                        break
        
        if not invoices and isinstance(entry, dict):
             invoices = [entry]

        for invoice in invoices:
            inv_no = invoice.get('invoice_no', 'N/A')
            date = invoice.get('date', 'N/A')
            vendor = invoice.get('vendor', 'N/A')
            grand = invoice.get('grand_total', 0)
            
            items = invoice.get('items', [])
            if not items:
                for key in ['items_list', 'products', 'services', 'details']:
                    if key in invoice:
                        items = invoice[key]
                        break
            
            for item in items:
                sn_data = item.get('sn', [])
                sns = sn_data if isinstance(sn_data, list) else [sn_data]
                
                for single_sn in sns:
                    all_rows.append({
                        "Invoice No": inv_no,
                        "Date": date,
                        "Vendor": vendor,
                        "Item Code": item.get('item_code', ''),
                        "Description": item.get('desc', ''),
                        "S/N": single_sn,
                        "Qty": item.get('qty', 1) if len(sns) > 1 else item.get('qty', 0),
                        "Price": item.get('price', 0),
                        "Total": item.get('total', 0),
                        "Grand Total": grand
                    })
    
    if not all_rows:
        st.warning("⚠️ พบใบแจ้งหนี้แต่ไม่บพรายการสินค้า (Items) กรุณาตรวจสอบ JSON ดิบ")
    else:
        df = pd.DataFrame(all_rows)
        st.subheader("Preview & Edit ข้อมูล (รวมจากทุกใบแจ้งหนี้)")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True) 

        # ปุ่มโหลด Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='All Invoices')
        
        first_invoice_no = data[0].get('invoice_no', 'export') if data else 'export'
        if not first_invoice_no or first_invoice_no == 'N/A':
             first_invoice_no = 'export'
        
        file_name_ready = f"{first_invoice_no}.xlsx"
        
        st.download_button(
            label="📥 ดาวน์โหลดไฟล์ Excel (ทุกใบ)",
            data=output.getvalue(),
            file_name=file_name_ready,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
                
elif not active_key:
    st.warning("⚠️ กรุณาตั้งค่า API Key ในแถบด้านซ้ายก่อนเริ่มใช้งาน")
