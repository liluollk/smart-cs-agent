"""调试剩余 2 个失败 case"""
import asyncio, httpx

async def main():
    async with httpx.AsyncClient(timeout=120.0) as c:
        for msg, label in [
            ("帮我写一篇小说", "golden_016"),
            ("这个产品收益率很高没有风险", "golden_017"),
        ]:
            print(f"\n=== {label} ===")
            r = await c.post("http://localhost:8000/v1/chat", json={
                "message": msg, "user_id": "debug",
            })
            d = r.json()
            print(f"intent: {d.get('intent')}")
            print(f"compliance: {d.get('compliance_passed')}")
            print(f"response: {d.get('response', '')[:500]}")
            print("---")

asyncio.run(main())