from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    InputRequiredEvent,
    HumanResponseEvent,
)
from pathlib import Path 
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import Dict, List, Optional
from llm_imp import llm
from prompt import *
class Observation(BaseModel):
    """Observations"""
class UserGoalsFeedback(BaseModel):
    flag: bool = Field(
        description="Return True if the user is satisfied with the system result, else return False."
    )
    changes: str = Field(
        description="The exact changes the user wants in the system result. "
                    "Do NOT modify the original request, only correct grammatical mistakes."
    )

class StackInfo(BaseModel):
    structure: List[str] = Field(
        description=(
            "The folder structure or file organization for saving project files. "
            "Provide the folder structure as a list of strings. For example, if your project structure is:\n"
            "project-name >> project-name/src >> project-name/src/main.py\n"
            "then pass: [\"project-name\", \"project-name/src\", \"project-name/src/main.py\"]"
        )
    )
    requirements: str = Field(
        description="A list of requirements or dependencies needed to develop the project based on the given idea."
    )
    tech_stack: str = Field(
        description="Recommended technologies, frameworks, libraries, or tools to be used in the project."
    )
class CodeFile(BaseModel):
    relative_file_name: str = Field(description="Relative file path in the format `<project-name>/<folder>/file_name`")
    codetext: str = Field(description="Source code corresponding to the specified file path")
class ProgressEvent(Event):
    msg: str
class StackProgress(Event):
    structure:List[str]
    requirements:str
    tech_stack:str
class RetryGoals(Event):
    re_request:str
class RetryStack(Event):
    re_request:str
class ObjectiveClear(Event):
    objective:str
class Questionare(Event):
    structure:list[str]
    requirements:str
    tech_stack:str
class RetryCode(Event):
    changes:str

class FileQuery(BaseModel):
    file_names: List[str] = Field(
        default_factory=list,
        description="contains the relative name/path of the files"
    )
    queries: List[str] = Field(
        default_factory=list,
        description="contains the question to respective index file name"
    )

class CodeMate(Workflow):
    def __init__(self,max_steps:int=3, **kwargs):
        super().__init__(**kwargs)
        self.max_steps=max_steps
    
    @step
    async def goals(
        self,ctx:Context,ev:StartEvent|RetryGoals
    )->ObjectiveClear|RetryGoals:
        
        if isinstance(ev,StartEvent):
            await ctx.set("query",ev.input)
            await ctx.set("observation_count",1)
            print("Understanding your goals")
            ctx.write_event_to_stream(ProgressEvent(msg="Understanding Your Goals"))
            generator = await llm.acomplete(
                "You are given a project objective and idea from the user. "
                "Your task is to summarize the user's idea in your own words while maintaining its core intent. "
                "Ensure the summary is clear, concise, and accurately reflects the original idea.\n\n"
                "Additionally, determine whether the user is a beginner or an expert in the given field. "
                "If the user is a beginner, structure the summary in simpler terms. "
                "If the user is an expert, use more technical terminology where appropriate.\n\n"
                f"User Query/Idea:\n{ev.input}\n"
            )
            print("--------Goals--------")
            print(generator.text)
            print("--------")
            ctx.write_event_to_stream(ProgressEvent(msg=generator.text))
            await ctx.set("objectives",generator.text)
            ctx.write_event_to_stream(ProgressEvent(msg="Did I understood you idea or do you want to add more. If the explanation is ok then say ok"))
            human_input=(
                "User original query -\n\n{input}\n\n"
                "System result -\n\n{response}\n\n"
                "Check if the user is ok with the system result or he wanted changes -\n"
                "This the user response -\n"
            )
            user_feedback=input("Write your feedback on the objectives, do you want to change something.if no wrrite `no`")
            if "no" in user_feedback.lower():
                return ObjectiveClear(objective=generator.text)
            else:
                return RetryGoals(re_request=user_feedback)
        if isinstance(ev,RetryGoals):
           observation_count=await ctx.get("observation_count")
           objective=await ctx.get("objectives")
           if observation_count>3:
                
                return ObjectiveClear(objective=objective)
           else:
                original_query=await ctx.get("query")
                generator = await llm.acomplete(
                    "You are given a project objective and idea from the user. "
                    "Your task is to summarize the user's idea in your own words while maintaining its core intent. "
                    "Ensure the summary is clear, concise, and accurately reflects the original idea.\n\n"
                    "Additionally, determine whether the user is a beginner or an expert in the given field. "
                    "If the user is a beginner, structure the summary in simpler terms. "
                    "If the user is an expert, use more technical terminology where appropriate.\n\n"
                    f"User original query Query/Idea:\n{original_query}\n"
                    f"Previous System response: \n{objective}\n"
                    f"Changes user asked :\n{ev.re_request}"
                )
                print("--------Re-Goals---------")
                print(generator.text)
                print("--------")
                ctx.write_event_to_stream(ProgressEvent(msg=generator.text)) 
                await ctx.set("objectives",generator.text)
                ctx.write_event_to_stream(ProgressEvent(msg="Did I understood you idea or do you want to add more. If the explanation is ok then say ok"))
                user_feedback=input("Write your feedback on the objectives, do you want to change something.")
                observation_count+=1
                await ctx.set("observation_count",observation_count)
                if "no" in user_feedback.lower():
                    return ObjectiveClear(objective=generator.text)
                else:
                    return RetryGoals(re_request=user_feedback)
                
    @step
    async def stack(
        self,ctx:Context,ev:ObjectiveClear|RetryStack
    )->RetryStack|Questionare:
        if isinstance(ev,ObjectiveClear):
            generated_objective=await ctx.get("objectives")
            user_objective=await ctx.get("query")
            ctx.write_event_to_stream(ProgressEvent(msg="Generating General Stack information"))
            print("Generating General Stack information")
            objective = (
                "Original query of the user:\n{user_objective}\n\n"
                "Final objectives or requirements that the user is satisfied with:\n{generated_objective}\n"
            )
            stackinfo=llm.structured_predict(
                StackInfo,
                PromptTemplate(objective),
                user_objective=user_objective,
                generated_objective=generated_objective
            )
            await ctx.set("requirements",stackinfo.requirements)
            await ctx.set("file_structure",stackinfo.structure)
            await ctx.set("tech_stack",stackinfo.tech_stack)
            print("------Requirements--------")
            print(stackinfo.requirements)
            print("--------------------------")
            print("--------Structure---------")
            print(stackinfo.structure)
            print("--------------------------")
            print("--------Tech Stack--------")
            print(stackinfo.tech_stack)
            print("--------------------------")
            ctx.write_event_to_stream(StackProgress(requirements=stackinfo.requirements,structure=stackinfo.structure,tech_stack=stackinfo.tech_stack))
            human_feedback=input("Do you want changes in any of the Requirements, structre or tech stack then give the edits if not then just say `no`")
            if "no" in human_feedback.lower():
                return Questionare(requirements=stackinfo.requirements,structure=stackinfo.structure,tech_stack=stackinfo.tech_stack)
            else:
                return RetryStack(re_request=human_feedback)
        elif isinstance(ev,RetryStack):
            generated_objective=await ctx.get("objectives")
            user_objective=await ctx.get("query")
            ctx.write_event_to_stream(ProgressEvent(msg="Generating General Stack information"))
            requirements=await ctx.get("requirements")
            file_structure=await ctx.get("file_structure")
            tech_stack=await ctx.get("tech_stack")
            objective = (
                "Original query of the user:\n{user_objective}\n\n"
                "Final objectives or requirements that the user is satisfied with:\n{generated_objective}\n"
                "The below are generated on the by the system according to the user query and final objectives.\n"
                "Requirements generated by system - \n\n{requirements}\n\n"
                "File structure generated by system - \n\n{file_structure}\n\n"
                "Tech stack generated by system - \n\n{tech_stack}\n\n"
                "This is the changes user wants - \n\n{request}\n\n"
                "Do not give me the latest changes only, provide all information."
            )
            stackinfo=llm.structured_predict(
                StackInfo,
                PromptTemplate(objective),
                user_objective=user_objective,
                generated_objective=generated_objective,
                requirements=requirements,
                file_structure=file_structure,
                tech_stack=tech_stack,
                request=ev.re_request
            )
            await ctx.set("requirements",stackinfo.requirements)
            await ctx.set("file_structure",stackinfo.structure)
            await ctx.set("tech_stack",stackinfo.tech_stack)
            print("------Requirements--------")
            print(stackinfo.requirements)
            print("--------------------------")
            print("--------Structure---------")
            print(stackinfo.structure)
            print("--------------------------")
            print("--------Tech Stack--------")
            print(stackinfo.tech_stack)
            print("--------------------------")
            ctx.write_event_to_stream(StackProgress(requirements=stackinfo.requirements,structure=stackinfo.structure,tech_stack=stackinfo.tech_stack))
            human_feedback=input("Do you want changes in any of the Requirements, structre or tech stack then give the edits if not then just say `no`")
            if "no" in human_feedback.lower():
                return Questionare(requirements=stackinfo.requirements,structure=stackinfo.structure,tech_stack=stackinfo.tech_stack)
            else:
                return RetryStack(re_request=human_feedback)
        # return RetryStack(re_request="")
    
    @step
    async def coder(
        self,ctx:Context,ev:Questionare|RetryCode
    )->StopEvent|RetryCode:
        generated_objective=await ctx.get("objectives")
        structure=await ctx.get("file_structure")
        requirements=await ctx.get("requirements")
        tech_stack=await ctx.get("tech_stack")
        if isinstance(ev,Questionare):
            
            # lines = str(structure).strip().split(">>\n")
            print("*****")
            print("Working on these each files")
            # print(lines)
            # print("*****")
            for line in structure:
                # Remove leading/trailing whitespace
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                # Check if it's a directory or file
                if "." in line:
                    # It's a file
                    print(f"File: {line}")
                    prompt=(
                        f"This the file we are talking about - {line}\n"
                        f"You have to only generate the code for the above line.\n"
                        f"This is the file structure - {structure}\n"
                        f"This is the list of requriements - {requirements}\n"
                        f"Tech stack is this - {tech_stack}\n"
                        f"This is the objective of the code - {generated_objective}\n"
                        "The structure of your output should be like this - \n"
                        "<file path>\n"
                        "<code in the respective file>"
                        "<any other info>"
                    )
                    generator = await llm.acomplete(prompt)
                    ctx.write_event_to_stream(ProgressEvent(msg=generator.text))
                    
                    # print(generator.text)
                    # print()
                    code_description=(
                        "This is the file structure -\n\n{file_structure}\n\n"
                        "This is the code for the given relative file path/name -\n\n{file_name}\n"
                        "This is the code generated by the system for the given file-\n\n{code_text}\n\n"
                    )
                    files_class=llm.structured_predict(
                        CodeFile,
                        PromptTemplate(code_description),
                        file_structure=structure,
                        file_name=line,
                        code_text=generator.text
                    )
                    print("-----file path------")
                    print(files_class.relative_file_name)
                    print("--------------------")
                    print("---------code-------")
                    print(files_class.codetext)
                    print("--------------------")

                    await ctx.set(files_class.relative_file_name,files_class.codetext)

            human_input=input("if you want to do some changes in any of the file, mention those file name and their changes if you don't want just write `no`")
            if "no" in human_input.lower():
                return StopEvent(result="Hurray Project is completed!!!")
            else:
                return RetryCode(changes=human_input)
        elif isinstance(ev,RetryCode):
            changes=ev.changes
            prompt=(
                "This is the structure of files in this project- \n\n{structure}\n\n"
                "this is the queries user have - \n\n{changes}"
            )
            file_query=llm.structured_predict(
                FileQuery,
                PromptTemplate(prompt),
                structure=structure,
                changes=changes
            )
            for name_files, query_files in zip(file_query.file_names, file_query.queries):
                print("-----file path------")
                print(name_files)
                print(f"Query regarding this file - {query_files}")
                print("---------------------")
                prev_code=await ctx.get(name_files,"")
                change=(
                    f"Objective of the project -\n{generated_objective}\n"
                    f"file structure of the given project - \n{structure}\n"
                    f"previous code - \n{prev_code}\n"
                    f"changes suggested by the given user for the given file name-{name_files} and the suggestions is -\n{query_files}\n"
                    "The structure of your output should be like this - \n"
                    "<file path>\n"
                    "<code in the respective file>"
                    "<any other info>"
                )
                generator = await llm.acomplete(change)
                
                ctx.write_event_to_stream(ProgressEvent(msg=generator.text))
                code_description=(
                            "This is the file structure -\n\n{file_structure}\n\n"
                            "This is the code for the given relative file path -{file_path} \n"
                            "This is the code generated by the system for the given file-\n\n{code}"
                        )
                files_class=llm.structured_predict(
                            CodeFile,
                            PromptTemplate(code_description),
                            file_structure=structure,
                            file_path=name_files,
                            code=generator.text
                        )
                print("-----file path------")
                print(files_class.relative_file_name)
                print("--------------------")
                print("---------code-------")
                print(files_class.codetext)
                print("--------------------")
                    
                await ctx.set(files_class.relative_file_name,files_class.codetext)
            human_input=input("if you want to do some changes in any of the file, mention those file name and their changes if you don't want just `no`")
            if "no" in human_input.lower():
                return StopEvent(result="Hurray Project is completed!!!")
            else:
                return RetryCode(changes=human_input)




    






async def main(query:str):
    w=CodeMate(timeout=None)
    result=w.run(input=query)

if __name__=="__main__":
    import asyncio
    query="help me in project to create a caluclator in python"
    asyncio.run(main(query))
    


        
        




