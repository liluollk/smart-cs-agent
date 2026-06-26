"""快速调试脚本 — 单个 case 测试"""
import asyncio
import httpx


async def main():
    async with httpx.AsyncClient(timeout=120.0) as c:
        for msg, label in [
            ("我要退款，刚买的保险不想要了", "工单创建"),
            ("怎么开户，需要准备什么材料", "开户咨询"),
            ("理财产品A的收益率是多少", "知识检索"),
        ]:
            print(f"\n=== {label} ===")
            try:
                r = await c.post("http://localhost:8000/v1/chat", json={
                    "message": msg, "user_id": "debug",
                })
                d = r.json()
                print(f"intent: {d.get('intent')}")
                print(f"compliance: {d.get('compliance_passed')}")
                print(f"response: {d.get('response', '')[:300]}")
            except Exception as e:
                print(f"ERROR: {e}")

asyncio.run(main())