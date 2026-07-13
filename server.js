const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

// CONFIGURATION
const HF_TOKEN = process.env.HF_TOKEN || "YOUR_HUGGINGFACE_TOKEN_HERE"; 
const API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-1.5B-Instruct";

// Base route check karne ke liye ki server chal raha hai ya nahi
app.get('/', (req, res) => {
    res.json({ status: "NexaAI Server is Live!" });
});

// Main API Route jahan aapka app request bhejega
app.post('/ask', async (req, res) => {
    const userQuestion = req.body.question;
    
    if (!userQuestion) {
        return res.status(400).json({ error: "Question missing hai boss!" });
    }

    const systemPrompt = "You are NexaAI, a premium and smart AI assistant. Always reply in clear, natural Hinglish or Hindi. If the user asks about medicines, provide a helpful general answer but strictly add a mandatory disclaimer: 'Kripya koi bhi dava lene se pehle doctor se salah zaroori lein.'";
    const fullPrompt = `<|im_start|>system\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userQuestion}<|im_end|>\n<|im_start|>assistant\n`;

    try {
        const response = await axios.post(API_URL, {
            inputs: fullPrompt,
            parameters: { max_new_tokens: 350, temperature: 0.7 }
        }, {
            headers: { 'Authorization': `Bearer ${HF_TOKEN}` }
        });

        const rawText = response.data[0].generated_text;
        const nexaAnswer = rawText.split("<|im_start|>assistant\n").pop().replace("<|im_end|>", "").trim();
        
        // Response send karna aapke app ko
        res.json({ success: true, response: nexaAnswer });

    } catch (error) {
        res.status(500).json({ success: false, error: "Server busy hai ya Token issue hai." });
    }
});

// Port configuration Render/Glitch free servers ke liye
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`NexaAI active on port ${PORT}`));
