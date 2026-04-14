import streamlit as st
import pandas as pd
import google.genai as genai
from io import StringIO
import json
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="Gemini Function Calling Demo", layout="wide")

# --- MOCK DATABASE ---
INVENTORY_CSV = """product_id,product_name,quantity,warehouse,reorder_point,last_updated
SKU-12345,Widget Pro,1523,ALL,2000,2027-01-20T10:30:00Z
SKU-12345,Widget Pro,892,DAL,2000,2027-01-20T09:15:00Z
SKU-12345,Widget Pro,631,CHI,2000,2027-01-20T10:30:00Z
SKU-67890,Gadget Plus,5420,ALL,1000,2027-01-19T14:00:00Z
SKU-67890,Gadget Plus,3200,DAL,1000,2027-01-19T14:00:00Z
SKU-67890,Gadget Plus,2220,CHI,1000,2027-01-19T12:30:00Z
SKU-11111,Basic Unit,187,ALL,500,2027-01-20T08:45:00Z
SKU-11111,Basic Unit,187,CHI,500,2027-01-20T08:45:00Z
SKU-99999,Premium Pack,0,ALL,100,2027-01-18T16:30:00Z
SKU-99999,Premium Pack,0,DAL,100,2027-01-18T16:30:00Z"""

inventory_df = pd.read_csv(StringIO(INVENTORY_CSV))

# --- TOOL IMPLEMENTATIONS ---
def get_inventory(product_id: str, warehouse: str = None) -> dict:
    if warehouse:
        filtered = inventory_df[(inventory_df["product_id"] == product_id) & (inventory_df["warehouse"] == warehouse)]
    else:
        filtered = inventory_df[(inventory_df["product_id"] == product_id) & (inventory_df["warehouse"] == "ALL")]

    if len(filtered) == 0:
        return {"error": f"Product {product_id} not found"}
    
    row = filtered.iloc[0]
    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "quantity": int(row["quantity"]),
        "warehouse": warehouse or "ALL",
        "reorder_point": int(row["reorder_point"]),
        "last_updated": row["last_updated"]
    }

def calculator(expression: str) -> dict:
    try:
        # Note: In production, use a safe parser like 'numexpr'
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

# --- TOOL DEFINITIONS ---
tools_config = [
    {
        "function_declarations": [
            {
                "name": "get_inventory",
                "description": "Get current inventory level for a product. Returns quantity and warehouse location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product SKU (e.g., 'SKU-12345')"},
                        "warehouse": {"type": "string", "description": "Warehouse code (e.g., 'DAL', 'CHI'). Omit for total."}
                    },
                    "required": ["product_id"]
                }
            },
            {
                "name": "calculator",
                "description": "Perform mathematical calculations or comparisons.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression (e.g., '1523 < 2000')"}
                    },
                    "required": ["expression"]
                }
            }
        ]
    }
]

# --- STREAMLIT UI ---
st.title("📦 Gemini Inventory Assistant")
st.markdown("This app demonstrates **Function Calling** in practice. Gemini acts as the brain, querying our mock inventory database.")

# Sidebar for Setup
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Google AI API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    model_name = st.selectbox("Model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash-lite"])
    
    st.divider()
    st.subheader("Inventory Database (Live View)")
    st.dataframe(inventory_df[inventory_df["warehouse"] == "ALL"][["product_id", "product_name", "quantity", "reorder_point"]])

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Loop
if prompt := st.chat_input("Ask about stock levels (e.g., 'Should we reorder SKU-12345?')"):
    if not api_key:
        st.error("Please provide a Google AI API Key in the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gemini Logic
        try:
            client = genai.Client(api_key=api_key)
            # Create a new chat session for this interaction if needed, or maintain one
            if "chat" not in st.session_state:
                st.session_state.chat = client.chats.create(model=model_name, config={"tools": tools_config})
            
            chat = st.session_state.chat
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Initial send
                response = chat.send_message(prompt)
                
                # Handle potential tool calls in a loop
                while True:
                    function_calls = response.function_calls
                    if not function_calls:
                        break
                    
                    function_responses = []
                    for fc in function_calls:
                        fn_name = fc.name
                        fn_args = dict(fc.args)
                        
                        st.info(f"🛠️ Tool Call: `{fn_name}({fn_args})`")
                        
                        if fn_name == "get_inventory":
                            result = get_inventory(**fn_args)
                        elif fn_name == "calculator":
                            result = calculator(**fn_args)
                        else:
                            result = {"error": "Unknown tool"}
                        
                        st.success(f"✅ Result: {result}")
                        
                        function_responses.append(
                            genai.types.Part.from_function_response(
                                name=fn_name,
                                response={"result": result}
                            )
                        )
                    
                    # Send tool results back
                    response = chat.send_message(function_responses)
                
                # Display final text response
                full_response = response.text
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")
