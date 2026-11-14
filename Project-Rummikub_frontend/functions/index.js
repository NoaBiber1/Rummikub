const { onRequest } = require("firebase-functions/v2/https");

const PRIMARY =
  "https://serverless.roboflow.com/matan-kronfeld/workflows/rummikub-v1-logic";
const FALLBACK =
  "https://serverless.roboflow.com/infer/workflows/matan-kronfeld/rummikub-v1-logic";

async function postWorkflow(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  return {
    status: response.status,
    text: await response.text(),
  };
}

exports.visionWorkflow = onRequest(
  {
    region: "us-central1",
    cors: true,
    timeoutSeconds: 60,
    memory: "512MiB",
    maxInstances: 10,
    invoker: "public",
  },
  async (req, res) => {
    if (req.method === "OPTIONS") {
      res.status(204).send("");
      return;
    }
    if (req.method !== "POST") {
      res.status(405).json({ detail: "POST only" });
      return;
    }

    const payload =
      typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
    const apiKey = process.env.ROBOFLOW_API_KEY || payload.api_key;
    if (!apiKey) {
      res.status(400).json({ detail: "Missing Roboflow API key." });
      return;
    }

    const inputs = payload.inputs && typeof payload.inputs === "object" ? payload.inputs : {};
    const image = inputs.image || payload.image;
    if (!image || typeof image !== "object") {
      res.status(400).json({ detail: "Missing image input." });
      return;
    }

    const body = JSON.stringify({
      api_key: apiKey,
      inputs: { image },
    });

    try {
      let result = await postWorkflow(PRIMARY, body);
      if (result.status === 404) {
        result = await postWorkflow(FALLBACK, body);
      }
      res.status(result.status);
      try {
        res.json(JSON.parse(result.text || "{}"));
      } catch (_) {
        res.json({ detail: (result.text || "").slice(0, 400) });
      }
    } catch (error) {
      res.status(502).json({
        detail: `Could not reach Roboflow: ${error.message}`,
      });
    }
  },
);
