import uvicorn

from voice_studio.app import app


def main():
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
