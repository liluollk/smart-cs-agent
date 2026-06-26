"""调试：看完整响应"""
import asyncio
import httpx


async def main():
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post("http://localhost:8000/v1/chat", json={
            "message": "我要退款，刚买的保险不想要了",
            "user_id": "debug_v2",
        })
        d = r.json()
        print("intent:", d.get("intent"))
        print("compliance:", d.get("compliance_passed"))
        print("FULL response:")
        print(d.get("response", ""))
        print("---END---")

asyncio.run(main())