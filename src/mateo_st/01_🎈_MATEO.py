import streamlit as st
from mateo_st import __version__ as mateo_version
from mateo_st.utils import cli_args, get_local_img, load_css, print_citation_info


def main():
    st.set_page_config(page_title="MATEO: MAchine Translation Evaluation Online", page_icon="💯")
    load_css("base")

    st.title("🎈 MATEO: MAchine Translation Evaluation Online")
    st.markdown(f'<div className="version"><p>v{mateo_version}</p></div>', unsafe_allow_html=True)

    st.markdown(
        "MAchine Translation Evaluation Online "
        "([MATEO](https://research.flw.ugent.be/projects/mateo-machine-translation-evaluation-online))"
        " brings automatic machine translation evaluation to the masses with"
        " an accessible user-interface. It was developed at Ghent University, in the"
        " [Language and Translation Technology Team (LT3)](https://lt3.ugent.be/) in 2022-2023."
        " It is currently canonically hosted at the [Dutch Language Institute](https://ivdnt.org/) (INT)" 
        " at [https://mateo.ivdnt.org/](https://mateo.ivdnt.org/)."
    )

    st.markdown(
        "MATEO was built to cater to both experts and non-experts. Users can be system builders, MT users and"
        " researchers, and also people from Social Sciences and Humanities (SSH), as well as teachers and students."
        " As such, MATEO can play a crucial role in research _and_ education by streamlining and"
        " simplifying the evaluation aspect of MT research on the one hand and enhancing digital literacy on the other."
    )

    st.markdown(
        "Because data security and privacy should not be something for you to worry about, **MATEO does not store"
        " your data**. All processing happens in-memory which means that as soon as processing is done your data"
        " is off our systems."
    )

    st.markdown("## ✒️ Citing MATEO")

    st.markdown("If you use MATEO for your work, please cite the following reference.")

    print_citation_info(expander_text=None)

    st.markdown("## 🏆 Funding")

    st.markdown(
        "MATEO was funded by a starting grant from the [European Association for Machine Translation](https://eamt.org/),"
        " and a significant full grant from [CLARIN.eu](https://www.clarin.eu/)'s _Bridging Gaps_ initiative. The"
        " project ran from mid 2022 until the end of June 2023 but is still maintained in an"
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
        " These steps are discussed in more detail on [GitHub](https://github.com/BramVanroy/mateo-demo).\n"
        "There are three main options to run your own instance of MATEO:\n"
        "- **On Hugging Face Spaces**: MATEO is also [running](https://huggingface.co/spaces/BramVanroy/mateo-demo) on the free platform"
        " of 🤗 Hugging Face in a so-called 'Space'. If you have an account (free) on that platform, you can easily"
        " duplicate the running MATEO instance to your own profile by clicking"
        " [this link](https://huggingface.co/spaces/BramVanroy/mateo-demo?duplicate=true)."
        " That means that you can create a private duplication of the MATEO interface **just for you** and free of"
        " charge! If the link should not work, you can follow these steps:\n"
        "\t1. Go to the [Space](https://huggingface.co/spaces/BramVanroy/mateo-demo) (if you are not already there);\n"
        "\t2. in the top right (below your profile picture) you should click on the three vertical dots;\n"
        "\t3. choose 'Duplicate space', _et&nbsp;voilà!_, a new space should now be running on your own profile\n"
        "- **In Python**: you can also clone the repository and install MATEO locally with Python. This"
        " process is described in more detail on [GitHub](https://github.com/BramVanroy/mateo-demo);\n"
        "- **With Docker**: using Docker to spin up an instance of this website on your own device (laptop, computer,"
        " server). This option allows you to modify the Docker image and the options to your needs. More information on"
        " [GitHub](https://github.com/BramVanroy/mateo-demo).",
        unsafe_allow_html=True,
    )

    st.markdown("## ✒️Contact, issues, and collaboration")

    st.markdown(
        "Would you like additional functionality in this project? Do you want to collaborate? Or just want to get in"
        " touch? Give me a shout on Twitter/X [@BramVanroy](https://twitter.com/BramVanroy),"
        " [LinkedIn](https://www.linkedin.com/in/bramvanroy/), or"
        " [send an email](https://www.kuleuven.be/wieiswie/nl/person/00099027). I'd ❤️ to hear from you!"
    )
    st.markdown(
        "Do you have technical questions, suggestions or ideas? Want to collaborate or add something to MATEO?"
        " Then head over to the [GitHub issues page](https://github.com/BramVanroy/mateo-demo)!"
    )


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
