import streamlit as st
from mateo_st.utils import cli_args, get_local_img, load_css


def main():
    st.set_page_config(page_title="MATEO: MAchine Translation Evaluation Online", page_icon="💯")
    load_css("base")

    st.title("🎈 MATEO: MAchine Translation Evaluation Online")

    st.markdown(
        "MAchine Translation Evaluation Online "
        "([MATEO](https://research.flw.ugent.be/projects/mateo-machine-translation-evaluation-online))"
        " brings automatic machine translation evaluation to the masses with"
        " an accessible user-interface. It is being developed at Ghent University, in the"
        " [Language and Translation Technology Team (LT3)](https://lt3.ugent.be/)."
    )

    st.markdown(
        "MATEO was built to cater to both experts and non-experts. Users can be system builders, MT users and"
        " researchers, and also people from Social Sciences and Humanities (SSH), as well as teachers and students."
        " As such, MATEO can play a crucial role in research _and_ education by streamling and"
        " simplifying the evaluation aspect of MT research on the one hand and enhancing digital literacy on the other."
    )

    st.markdown("## 🏆 Funding")

    st.markdown(
        "MATEO is funded by a starting grant from the [European Association for Machine Translation](https://eamt.org/),"
        " and a significant full grant from [CLARIN.eu](https://www.clarin.eu/)'s _Bridging Gaps_ initiative. The"
        " project will officially run until July 2023 but will continue to be maintained afterwards in an"
        " [open-source manner](https://github.com/BramVanroy/mateo-demo/)."
    )

    eamt_col, _, clarin_col = st.columns((2, 1, 3))
    eamt_col.image(get_local_img("eamt.png"))
    clarin_col.image(get_local_img("clarin.png"))

    st.markdown("## ☁️Self-hosting")
    st.markdown(
        "This website is provided for free as a hosted application. That means that you, or anyone else, can use it."
        " The implication is that it is possible that the service will be slow depending on the usage of the system."
        " As such, specific attention was paid to making it easy for you to set up your own instance that you can use!"
        "There are two main options:\n"
        "- **Easy**: MATEO is also [running](https://huggingface.co/spaces/BramVanroy/mateo-demo) on the free platform"
        " of 🤗 Hugging Face in a so-called 'Space'. If you have an account (free) on that platform, you can easily"
        " duplicate this instance to your own profile. That means that you can create a private duplication of the"
        " MATEO interface **just for you** and free of charge!\n"
        "\t1. You can do this by going to the"
        " [Space](https://huggingface.co/spaces/BramVanroy/mateo-demo) (if you are not already there);\n"
        "\t2. in the top right (below your profile picture) you should click on the three vertical dots;\n"
        "\t3. choose 'Duplicate space', _et&nbsp;voilà!_\n\n"
        "If you have questions about setting this up [get in touch](#contact-issues-and-collaboration).\n"
        "- **Advanced**: using Docker to spin up an instance of this website on your own device (laptop, computer,"
        " server). This option allows you to modify the Docker image and the options to your needs. More information on"
        " [GitHub](https://github.com/BramVanroy/mateo-demo).",
        unsafe_allow_html=True
    )

    st.markdown("## ✒️Contact, issues, and collaboration")

    st.markdown(
        "Would you like additional functionality in this project? Do you want to collaborate? Or just want to get in"
        " touch? Give me a shout on Twitter [@BramVanroy](https://twitter.com/BramVanroy) or"
        " [send an email](https://research.flw.ugent.be/nl/bram.vanroy). I'd ❤️ to hear from you!"
    )
    st.markdown(
        "Do you have technical questions, suggestions or ideas? Head over to the"
        " [Github issues page](https://github.com/BramVanroy/mateo-demo)."
    )


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
