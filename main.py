import streamlit as st

# Center the title using markdown with HTML
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://iee.eng.kmutnb.ac.th/iee/wp-content/uploads/2023/01/Master-Logo_KMUTNB_ENG-01.png' style='width: 250px; height: 250px; object-fit: cover;'>
        <br>
        <h3>40613701 Intelligent Systems Section 5</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Center the rest of the content with markdown and HTML
st.markdown(
    """
    <div style='text-align: center;'>
        <br><br><br>
        <strong>Presented by</strong>
        <br><br>
        6504062636098 Mr. Thanatsaen Kaoian
        <br><br><br><br><br>
        <strong>By</strong>
        <br><br>
        Asst.Prof. Thattapon Surasak
        <br><br><br><br><br>
        <strong>Semester 2, Academic Year 2024, King Mongkut's University of Technology North Bangkok </strong>
    </div>
    """,
    unsafe_allow_html=True
)