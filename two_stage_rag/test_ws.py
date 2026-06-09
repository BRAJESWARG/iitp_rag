import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/api/ws/query"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            query = "What is a transformer model?"
            print(f"Sending query: '{query}'")
            await websocket.send(json.dumps({"query": query}))
            
            print("\n--- Receiving Events ---")
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                type_ = data.get("type")
                if type_ == "stage":
                    print(f"\n[Stage Update] Stage: {data.get('stage')}")
                    if "stage1_candidates" in data:
                        print(f"  Stage 1 Candidates: {data['stage1_candidates']}")
                    if "stage2_top_docs" in data:
                        print(f"  Stage 2 Top Docs: {data['stage2_top_docs']}")
                        print(f"  Top Scores: {data['top_doc_scores']}")
                elif type_ == "chunk":
                    print(data.get("text"), end="", flush=True)
                elif type_ == "done":
                    print(f"\n\n[Done] Completed in {data.get('duration_ms')}ms using {data.get('model_used')}")
                    break
                elif type_ == "error":
                    print(f"\n[Error] {data.get('message')}")
                    break
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
