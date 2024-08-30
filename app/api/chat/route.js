import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import Groq from "groq-sdk";
import dotenv from "dotenv";

dotenv.config(".env.local");

const systemPrompt = `You are an AI assistant for a RateMyProfessor-style service, designed to help students find professors based on their queries. Your primary function is to provide information about the top 3 most relevant professors for each user question, using a Retrieval-Augmented Generation (RAG) system.

For each user query:
1. Analyze the question to understand the student's requirements, preferences, or concerns.
2. Use the RAG system to retrieve information about the 3 most relevant professors based on the query.
3. Present information about these professors in a clear, concise manner.
4. Include key details such as:
   - Professor's name
   - Subject/department
   - Overall rating (usually out of 5 stars)
   - A brief summary of student feedback
   - Any standout characteristics or teaching styles mentioned in reviews

5. If the query is vague or could be interpreted in multiple ways, ask for clarification before providing recommendations.
6. If fewer than 3 relevant professors are found, explain this and provide information on those available.
7. Avoid making personal judgments or recommendations. Instead, present the information objectively based on the retrieved data.
8. If asked about specific aspects (e.g., difficulty, workload), focus on those in your response.
9. Always maintain a helpful and informative tone, keeping in mind that your purpose is to assist students in making informed decisions about their education.

Remember: Your responses should be based solely on the information retrieved by the RAG system. Do not invent or assume information about professors or courses. If certain information is not available, simply state that it's not provided in the current data.

Respond to the user's query now, providing information on the top 3 most relevant professors based on their question.`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pc.index("rag").namespace("ns1");
  const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
  const { pipeline } = await import("@xenova/transformers");
  const extractor = await pipeline("feature-extraction", "Xenova/gte-small");

  const text = data[data.length - 1].content;
  const result = await extractor(text, { pooling: "mean", normalize: true });
  const embedding = Array.from(result.data);

  // Use the embedding for your Pinecone query
  const results = await index.query({
    topK: 5,
    includeMetadata: true,
    vector: embedding,
  });

  let resultString = "";
  results.matches.forEach((match) => {
    resultString += `
    Returned Results:
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n`;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

  const completion = await groq.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "llama3-8b-8192",
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });
  return new NextResponse(stream);
}
