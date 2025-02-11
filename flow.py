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
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
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
    structure: str = Field(
        description="The folder structure or file organization for saving project files."
    )
    requirements: str = Field(
        description="List of requirements needed to develop the project based on the given idea."
    )
    tech_stack: str = Field(
        description="Recommended technologies, frameworks, or tools to be used in the project."
    )

class ProgressEvent(Event):
    msg: str
class StackProgress(Event):
    structure:str
    requirements:str
    tech_stack:str
class RetryGoals(Event):
    re_request:str
class RetryStack(Event):
    re_request:str
class ObjectiveClear(Event):
    objective:str
class Questionare(Event):
    structure:str
    requirements:str
    tech_stack:str

class CodeMate(Workflow):
    def __init__(self,max_steps:int=3, **kwargs):
        super().__init__(**kwargs)
        self.llm=llm
        self.max_steps=max_steps
    
    @step
    async def goals(
        self,ctx:Context,ev:StartEvent|RetryGoals
    )->ObjectiveClear|RetryGoals:
        
        if isinstance(ev,StartEvent):
            await ctx.set("query",ev.input)
            await ctx.set("observation_count",1)
            ctx.write_event_to_stream(ProgressEvent(msg="Understanding Your Goals"))
            generator = await self.llm.astream_complete(
                "You are given a project objective and idea from the user. "
                "Your task is to summarize the user's idea in your own words while maintaining its core intent. "
                "Ensure the summary is clear, concise, and accurately reflects the original idea.\n\n"
                "Additionally, determine whether the user is a beginner or an expert in the given field. "
                "If the user is a beginner, structure the summary in simpler terms. "
                "If the user is an expert, use more technical terminology where appropriate.\n\n"
                f"User Query/Idea:\n{ev.input}\n"
            )
            async for response in generator:
            # Allow the workflow to stream this piece of response
                ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
            await ctx.set("objectives",str(response))
            ctx.write_event_to_stream(ProgressEvent(msg="Did I understood you idea or do you want to add more. If the explanation is ok then say ok"))
            human_input=(
                "User original query -\n\n{input}\n\n"
                "System result -\n\n{response}\n\n"
                "Check if the user is ok with the system result or he wanted changes -\n"
                "This the user response -\n"
            )
            user_feedback=input("Write your feedback on the objectives, do you want to change something.if no wrrite `no`")
            # check=self.llm.structured_predict(
            #     UserGoalsFeedback,
            #     PromptTemplate(human_input+user_feedback),
            #     input=ev.input,
            #     response=str(response),
            # )
            if "no" in user_feedback.lower():
                return ObjectiveClear(objective=str(response))
            else:
                return RetryGoals(re_request=user_feedback)
        if isinstance(ev,RetryGoals):
           observation_count=await ctx.get("observation_count")
           objective=await ctx.get("objectives")
           if observation_count>3:
                
                return ObjectiveClear(objective=objective)
           else:
                original_query=await ctx.get("query")
                generator = await self.llm.astream_complete(
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
                async for response in generator:
                # Allow the workflow to stream this piece of response
                    ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
                await ctx.set("objectives",str(response))
                ctx.write_event_to_stream(ProgressEvent(msg="Did I understood you idea or do you want to add more. If the explanation is ok then say ok"))
                user_feedback=input("Write your feedback on the objectives, do you want to change something.")
                # check=self.llm.structured_predict(
                #     UserGoalsFeedback,
                #     PromptTemplate(human_input+user_feedback),
                #     input=ev.input,
                #     response=str(response),
                # )
                observation_count+=1
                await ctx.set("observation_count",observation_count)
                if "no" in user_feedback.lower():
                    return ObjectiveClear(objective=str(response))
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
            objective = (
                "Original query of the user:\n{user_objective}\n\n"
                "Final objectives or requirements that the user is satisfied with:\n{generated_objective}\n"
            )
            stackinfo=self.llm.structured_predict(
                StackInfo,
                PromptTemplate(objective),
                user_objective=user_objective,
                generated_objective=generated_objective
            )
            await ctx.set("requirements",stackinfo.requirements)
            await ctx.set("file_structure",stackinfo.structure)
            await ctx.set("tech_stack",stackinfo.tech_stack)
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
            stackinfo=self.llm.structured_predict(
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
            ctx.write_event_to_stream(StackProgress(requirements=stackinfo.requirements,structure=stackinfo.structure,tech_stack=stackinfo.tech_stack))
            human_feedback=input("Do you want changes in any of the Requirements, structre or tech stack then give the edits if not then just say `no`")
            if "no" in human_feedback.lower():
                return Questionare(requirements=stackinfo.requirements,structure=stackinfo.structure,tech_stack=stackinfo.tech_stack)
            else:
                return RetryStack(re_request=human_feedback)
        
    






async def main():

           


        
        




