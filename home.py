import streamlit as st


def main_page():
    st.set_page_config(
        page_title="Paper research and slide generation",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",  # expand side bar (horizontally)
    )
    # set_streamlit_page_config_once()
    # st.title("Paper research and slide generation")

    main_page = st.Page("pages/main_page.py", title="🏠 Home")
    slide_gen_page = st.Page(
        "pages/slide_generation.py", title="🧾 Slide Generation"
    )
    # chat_page = st.Page("app_pages/chat.py", title="💬 Chat")
    pg = st.navigation([main_page, slide_gen_page])

    pg.run()


if __name__ == "__main__":
    main_page()