const express = require('express')
const cors = require('cors')
const bodyParser = require('body-parser')
require('dotenv').config();

const { Configuration, OpenAIApi } = require('openai')

const config = new Configuration({
    apiKey: process.env.GPT_API_KEY,
    organization: process.env.GPT_ORG_KEY
})

const openai = new OpenAIApi(config)
//server
const app = express()
app.use(bodyParser.json())
app.use(cors())

//endpoint chatGPT
var messages = []
app.post('/chat', async (req, res) => {
    const msg = req.body.msg;
    messages.push(msg)
    const completion = await openai.createChatCompletion({
        model: "gpt-3.5-turbo",
        messages: messages,
        max_tokens: 512,
        temperature: 0.1
    })
    messages.push(completion.data.choices[0].message)
    res.send(completion.data.choices[0].message.content)
})

const port = 8080
app.listen(port, () => {
    console.log(`Server listening on port ${port}`)
})