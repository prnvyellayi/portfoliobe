const express = require('express')
const cors = require('cors')
const bodyParser = require('body-parser')
const { ConversationalRetrievalQAChain } = require("langchain/chains");
const { BufferMemory } = require('langchain/memory')
const { ChatOpenAI } = require("langchain/chat_models/openai");
const { OpenAIEmbeddings } = require('langchain/embeddings/openai')
const { MemoryVectorStore } = require('langchain/vectorstores/memory')
const { TextLoader } = require('langchain/document_loaders/fs/text')
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter')
require('dotenv').config();

const memory = new BufferMemory({
    memoryKey: "chat_history",
    returnMessages: true,
});

//server
const app = express()
app.use(bodyParser.json())
app.use(cors())

// //endpoint chatGPT
app.post('/chat', async (req, res) => {
    const msg = req.body.msg.content;
    // console.log(msg)
    
    const loader = new TextLoader('./resume.txt')
    const docs = await loader.load()
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 0,
    });

    const splitDocs = await textSplitter.splitDocuments(docs);
    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.GPT_API_KEY,
        temperature: 0
    })
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
        openAIApiKey: process.env.GPT_API_KEY,
        temperature: 0
    });
    const chain = ConversationalRetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
        memory
    });

    const response = await chain.call({
        question: msg
    });
    res.send(response.text)
})

const port = 8080
app.listen(port, () => {
    console.log(`Server listening on port ${port}`)
})