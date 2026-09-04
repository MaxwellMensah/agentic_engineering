from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

gemini_model = LLM(
    model="gemini/gemini-2.5-flash",  # Always include the 'gemini/' prefix
    temperature=0.7,
)

# Define agents with roles and backstories
researcher = Agent(
    role="Senior Technical Researcher",
    goal="Gather accurate, structured insights on technical concepts",
    backstory="An expert researcher capable of analyzing emerging software and physics trends.",
    verbose=True,  # Logs agent-level thinking
    llm=gemini_model,
)

writer = Agent(
    role="Technical Writer",
    goal="Synthesize research into structured, accessible articles",
    backstory="A seasoned technology journalist who transforms dense research into clear insights.",
    verbose=True,  # Logs agent-level thinking
    llm=gemini_model,
)

# Define explicit tasks
research_task = Task(
    description="Investigate the impact of Quantum Computing on modern cryptography.",
    expected_output="Bullet points of key findings and vulnerability timelines.",
    agent=researcher,
)

write_task = Task(
    description="Using the research findings, draft a 2-paragraph summary report.",
    expected_output="A well-structured 2-paragraph final report.",
    agent=writer,
)

# Assemble and execute the Crew pipeline
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,  # Enables crew-level pipeline logging in terminal
    output_log_file="trace.log",  # Saves the step-by-step trace to trace.log
)

result = crew.kickoff()

print("\n================ FINAL REPORT ================\n")
print(result)
