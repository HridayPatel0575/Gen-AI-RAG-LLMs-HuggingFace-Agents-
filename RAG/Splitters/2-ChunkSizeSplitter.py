from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
The morning sky carried a quiet softness that made the city feel slower than usual. Streets that were normally filled with impatient horns and hurried footsteps appeared calm, as if time itself had decided to pause for a moment. A vendor arranged fruits carefully, aligning colors more than shapes, while a stray dog slept peacefully under the shade of a closed shop. Somewhere in the distance, a train horn echoed, reminding everyone that movement would eventually resume.

In a small room above a bookstore, a ceiling fan rotated lazily, pushing warm air from one corner to another. Papers lay scattered on a wooden desk, covered in notes written with different pens at different times. Some sentences were confident, others scratched out entirely, showing how ideas often change before finding their final form. A cup of cold tea stood forgotten, its surface reflecting light from the nearby window.

Stories do not always begin with grand events. Sometimes they start with observation, with noticing how shadows stretch longer in the evening or how silence can be louder than conversation. Every place carries layers of memory, even when the people who created them have moved on. A cracked wall may look ordinary, yet it has witnessed laughter, arguments, dreams, and disappointments.

Technology has transformed how humans interact with the world, but it has not changed the core desire to understand and be understood. Messages now travel instantly, crossing continents in seconds, yet the emotions behind them remain fragile and complex. A simple sentence can carry hope, confusion, or relief, depending on who reads it and when. Machines process language efficiently, but meaning still depends on context.

In parks, benches act as temporary resting points for strangers who share nothing except a moment of stillness. One person scrolls endlessly on a phone, another stares at trees as if searching for answers in the leaves. Children run without worrying about direction, while elders watch quietly, carrying stories that rarely get told. These small scenes repeat daily, unnoticed yet essential.

Ideas often arrive unexpectedly. They show up during routine walks, while washing dishes, or in the brief seconds before sleep. Capturing them requires attention, because once ignored, they fade quickly. Writing is less about perfection and more about preservation. It allows thoughts to exist outside the mind, where they can be examined, challenged, and reshaped.

Learning is rarely linear. Progress includes pauses, mistakes, and moments of doubt. Understanding deepens when questions replace assumptions. Whether studying science, art, or human behavior, curiosity acts as the driving force. Without it, information remains static. With it, even simple facts connect into larger patterns.

Across different cultures and regions, people find meaning in rituals, routines, and shared experiences. Food brings comfort, music brings connection, and stories bring identity. Though languages differ, emotions translate easily. Joy, fear, ambition, and regret are universal, expressed through countless forms yet deeply familiar.

As night approaches, lights flicker on in windows, each revealing a separate world behind glass. Some rooms are filled with conversation, others with solitude. The city resumes its rhythm, slightly altered but persistent. Tomorrow will look similar, yet not identical, shaped by countless small choices made today.

In the end, randomness itself has structure. What appears unplanned often follows hidden patterns. Text, like life, does not always need a clear purpose to exist. Sometimes it simply needs space to unfold, line by line, carrying fragments of thought into something quietly complete.
"""


splitter =RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)