import asyncio

from modules.upd_xai_bridget import UdpToXaiBridge


async def main() -> None:
    bridge = UdpToXaiBridge()
    await bridge.run()


if __name__ == "__main__":
    asyncio.run(main())