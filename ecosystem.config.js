module.exports = {
  apps: [
    {
      name: "igbo-rag-api",
      script: "/Users/kere/igbo-rag/.venv/bin/uvicorn",
      args: "api:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      cwd: "/Users/kere/igbo-rag",
      env: {
        PYTHONPATH: "/Users/kere/igbo-rag/src",
      },
      autorestart: true,
      max_restarts: 10,
      // FAISS index load is slow; give it room before pm2 considers startup failed
      min_uptime: "30s",
      kill_timeout: 10000,
      out_file: "/Users/kere/igbo-rag/logs/api.out.log",
      error_file: "/Users/kere/igbo-rag/logs/api.err.log",
      merge_logs: true,
      time: true,
    },
  ],
};
