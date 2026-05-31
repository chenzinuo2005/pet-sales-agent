"""CLI 调度器 — argparse 命令路由"""
import argparse
import os
import sys

# Windows: reconfigure stdout to UTF-8 to handle emoji/Chinese
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from app.common.logger import setup_logging, logger


def validate_env() -> None:
    """验证必填环境变量，缺失时 exit code 2"""
    required_vars = [
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DASHSCOPE_API_KEY",
        "TAVILY_API_KEY",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"缺少环境变量: {', '.join(missing)}, 请检查 .env")
        sys.exit(2)


def cmd_chat(args: argparse.Namespace) -> None:
    """交互对话模式"""
    validate_env()
    from app.agents.pet_agent import chat_with_agent

    thread_id = "default" if not args.new_session else None  # None → 生成新 UUID
    print("小宠: 你好！我是萌宠之家的智能客服小宠，有什么可以帮你的？(输入 /quit 退出)")

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("再见！")
            break

        image_path = None
        message = user_input

        # 解析 image: 前缀
        if user_input.startswith("image:"):
            parts = user_input[6:].strip()
            # 分离路径和问题
            first_space = -1
            for i, ch in enumerate(parts):
                if ch == " ":
                    first_space = i
                    break
            if first_space == -1:
                image_path = parts
                message = "这是什么品种？"
            else:
                image_path = parts[:first_space]
                message = parts[first_space + 1 :].strip()

            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                continue

        # 流式输出回复
        print("\n小宠: ", end="", flush=True)
        for token in chat_with_agent(message, image_path, thread_id):
            print(token, end="", flush=True)
        print()


def cmd_train_cnn(args: argparse.Namespace) -> None:
    """训练 CNN 模型"""
    from app.cnn.train import train

    train(args.data_root)


def cmd_evaluate_cnn(args: argparse.Namespace) -> None:
    """评估 CNN 模型"""
    from app.cnn.evaluate import evaluate_model

    logger.info(f"评估 CNN 模型, 数据集: {args.data_root}")
    test_loss, test_acc = evaluate_model(args.data_root)
    target_met = test_acc >= 0.85
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"85% target: {'PASS' if target_met else 'NOT MET'}")


def cmd_predict_cnn(args: argparse.Namespace) -> None:
    """单图品种预测 (仅 CNN，不进 Agent)"""
    from app.cnn.inference import predict_breed

    if not os.path.exists(args.image):
        logger.error(f"图片文件不存在: {args.image}")
        sys.exit(3)

    result = predict_breed(args.image)
    print(result.model_dump_json(indent=2))


def cmd_init_rag(args: argparse.Namespace) -> None:
    """初始化 RAG 向量库"""
    validate_env()
    from app.agents.rag_tools import init_vector_store

    logger.info("初始化 RAG 向量库...")
    vs, count = init_vector_store()
    logger.info(f"RAG 初始化完成，共 {count} 个文档片段")
    print(f"向量库已保存至 resources/chroma_db/ (共 {count} 个片段)")


def cmd_serve(args: argparse.Namespace) -> None:
    """启动 FastAPI 后端服务"""
    import uvicorn

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    logger.info(f"启动 API 服务: http://{host}:{port}")
    uvicorn.run("app.api.server:app", host=host, port=port, reload=args.reload)


def main() -> None:
    setup_logging()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(
        prog="pet",
        description="宠物销售智能客服助手",
    )
    sub = parser.add_subparsers(dest="command")

    # chat
    chat_parser = sub.add_parser("chat", help="交互对话")
    chat_parser.add_argument("--new-session", action="store_true", help="新会话")
    chat_parser.set_defaults(func=cmd_chat)

    # train-cnn
    train_parser = sub.add_parser("train-cnn", help="训练CNN模型")
    train_parser.add_argument("--data-root", required=True, help="Oxford Pets 数据集路径")
    train_parser.set_defaults(func=cmd_train_cnn)

    # evaluate-cnn
    eval_parser = sub.add_parser("evaluate-cnn", help="评估CNN模型")
    eval_parser.add_argument("--data-root", required=True, help="Oxford Pets 数据集路径")
    eval_parser.set_defaults(func=cmd_evaluate_cnn)

    # predict-cnn
    pred_parser = sub.add_parser("predict-cnn", help="单图品种预测")
    pred_parser.add_argument("--image", required=True, help="图片路径")
    pred_parser.set_defaults(func=cmd_predict_cnn)

    # init-rag
    init_parser = sub.add_parser("init-rag", help="初始化RAG向量库")
    init_parser.set_defaults(func=cmd_init_rag)

    # serve (default)
    serve_parser = sub.add_parser("serve", help="启动API服务")
    serve_parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    serve_parser.add_argument("--port", type=int, default=8000, help="端口")
    serve_parser.add_argument("--no-reload", dest="reload", action="store_false", help="禁用热重载")
    serve_parser.set_defaults(func=cmd_serve, reload=True)

    args = parser.parse_args()
    if args.command is None:
        # 默认: 启动 API 服务
        cmd_serve(argparse.Namespace(host="127.0.0.1", port=8000, reload=True))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
