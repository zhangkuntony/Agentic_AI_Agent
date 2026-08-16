from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

job_description = """
SPS-Software Engineer (m/w/d) im Maschinenbau
Glaston Germany GmbH
Neuhausen-Hamberg
Feste Anstellung
Homeoffice möglich, Vollzeit
Erschienen: vor 1 Tag
Glaston Germany GmbH logo
SPS-Software Engineer (m/w/d) im Maschinenbau
Glaston Germany GmbH
slide number 1slide number 2slide number 3
Glaston ist eine internationale Marke mit weltweit führenden Unternehmen, die für zukunftsweisende Maschinen, Anlagen, Systeme und Dienstleistungen in der Bearbeitung von Architektur-, Fahrzeug- und Displayglas steht.

Mit unserer über 50-jährigen Erfahrung am Standort Glaston Germany GmbH in Neuhausen bei Pforzheim verbessern und sichern wir nachhaltig die Produktivität unserer Kunden bei der Fertigung von Architekturglas. Diesen Erfolg verdanken wir unseren motivierten und engagierten Mitarbeitenden und wurden so zu einem der führenden Anbieter von automatisierten und kundenspezifischen Anlagen.

Der Umgang mit Software liegt dir im Blut und du möchtest bei einem Hidden Champion durchstarten?
Dein Faible für Softwarelösungen und dein Herz für unterschiedliche Technologien sind ideale Voraussetzungen, um Maschinen wieder zu alter Stärke zu verhelfen?
Du hast einen ausgeprägten Servicegedanken und Spaß an der Arbeit mit Kunden?

Dann komm zu Glaston! Wir suchen ab sofort für unseren Bereich Service Upgrades Verstärkung!

SPS-SOFTWARE ENGINEER (M/W/D) IM MASCHINENBAU

Als SPS-Software Engineer (m/w/d) im Maschinenbau sind deine Aufgabengebiete:
Ausarbeitung und Weiterentwicklung von Kundenaufträgen und Upgrade-Konzepten
Selbstständige und termingerechte Bearbeitung von Kundenprojekten und Bereitstellung der notwendigen Dokumente
Unterstützung des Inbetriebnahme- und Servicepersonals im Haus und beim Kunden vor Ort
Diese Anforderungen musst du mitbringen:
Qualifizierte technische Ausbildung: Techniker, Studium oder vergleichbare Qualifikation
Mehrjährige Berufserfahrung im Serviceumfeld, idealerweise im Maschinen- und Anlagenbau
Umfangreiche Kenntnisse in verschiedenen SPS-Programmiersprachen (z.B. S7Classic, TIA, Simotion)
Bei uns profitierst du von folgenden Benefits:
Exzellente Rahmenbedingungen (z.B. attraktives Gehaltsmodell, flexible Arbeitszeiten mit Gleitzeit und Homeoffice-Möglichkeiten)
Attraktives Arbeitsumfeld in idyllisch-ländlicher Lage
Umfangreiche Mobilitätsförderung (z.B. Ladestation für Elektroautos)
Wellbeing am Arbeitsplatz
"""

# # 1. 使用Python内置的templates
prompt_template = (
    "Given a job description, decide whether it suites a junior Java developer."
    "\nJOB DESCRIPTION:\n{job_description}\n"
)
result = model.invoke(prompt_template.format(job_description=job_description))
print(result.content)

# 2. 使用LangChain内置的templates
lc_prompt_template = PromptTemplate.from_template(prompt_template)
print(lc_prompt_template.invoke({"job_description": "fake_jd"}))

# 使用chain，并传入完整的json
lc_prompt_template = PromptTemplate.from_template(prompt_template)
chain = lc_prompt_template | model | StrOutputParser()
result = chain.invoke({"job_description": job_description})
print(result)

# 使用chain，直接传入job_description
result = chain.invoke(job_description)
print(result)


# 使用提示词占位符
msg_template = HumanMessagePromptTemplate.from_template(prompt_template)
msg_example = msg_template.format(job_description="fake_jd")

chat_prompt_template = ChatPromptTemplate.from_messages([SystemMessage(content="You are a helpful assistant."), msg_template])
chain = chat_prompt_template | model | StrOutputParser()
result = chain.invoke({"job_description": job_description})
print(result)

chat_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."),
     ("human", prompt_template)]
)
chain = chat_prompt_template | model | StrOutputParser()
result = chain.invoke({"job_description": job_description})
print(result)


chat_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."),
     ("placeholder", "{history}"),
     # MessagesPlaceholder("history"),
     ("human", prompt_template)]
)
print(chat_prompt_template.invoke("fake"))

print(len(chat_prompt_template.invoke({"job_description": "fake", "history": [("human", "hi!"), ("ai", "hi!")]}).messages))

examples = [
    {"question": "q1", "answer": "a1"},
    {"question": "q2", "answer": "a2"}
]
test_template = PromptTemplate.from_template("substituted a: {a}")

system_template = PromptTemplate.from_template("a: {a} b: {b}")
system_template_part = system_template.partial(
    a="a" # you also can provide a function here
)
print(system_template_part.invoke({"b": "b"}).text)

print(system_template_part.invoke({"b": "b"}).text == system_template_part.format(b="b"))

system_template_part1 = PromptTemplate.from_template("a: {a}")
system_template_part2 = PromptTemplate.from_template("b: {b}")
system_template = system_template_part1 + " " + system_template_part2
print(system_template.invoke({"a": "a", "b": "b"}).text)

system_prompt_template = PromptTemplate.from_template("a: {a} b: {b}")
chat_prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_prompt_template.template),
     ("human", "hi"),
     ("ai", "{c}")])

messages = chat_prompt_template.invoke({"a": "a", "b": "b", "c": "c"}).messages
print(len(messages))
print(messages[0].content)
print(messages[1].content)
print(messages[2].content)
