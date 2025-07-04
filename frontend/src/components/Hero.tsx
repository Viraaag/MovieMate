

const Hero = () => {
    const posterUrls = [
        
        "https://m.media-amazon.com/images/I/813dE2pH7XL._UF1000,1000_QL80_.jpg", // Matrix
        "https://m.media-amazon.com/images/M/MV5BOTgyOGQ1NDItNGU3Ny00MjU3LTg2YWEtNmEyYjBiMjI1Y2M5XkEyXkFqcGc@._V1_.jpg", // Inception
        "https://resizing.flixster.com/-XZAfHZM39UwaGJIFWKAE8fS0ak=/v3/t/assets/p10543523_p_v8_as.jpg", // Interstellar
        "https://m.media-amazon.com/images/M/MV5BNzA1Njg4NzYxOV5BMl5BanBnXkFtZTgwODk5NjU3MzI@._V1_FMjpg_UX1000_.jpg", // Blade Runner 2049
        "https://stat4.bollywoodhungama.in/wp-content/uploads/2018/07/Tumbbad1-322x460.jpg",
        "https://m.media-amazon.com/images/M/MV5BYjI0NDQzYmEtNzMwZC00ODA3LTgzZDYtZTk5ODZjY2Y2OTkzXkEyXkFqcGc@._V1_.jpg", // Tenet
        "https://img.mensxp.com/media/content/2024/Dec/2---credit----Mythri-Movie-Makers_675d00431bea9.jpeg?w=780&h=1246&cc=1",
        
        "https://m.media-amazon.com/images/M/MV5BODk4ZTgwZWYtNDMyYi00NTczLWI3NGYtMGVjZmQ3MDUzZDliXkEyXkFqcGc@._V1_.jpg",
        "https://assets-prd.ignimgs.com/2022/01/26/thebatman-newbutton-1643232430643.jpg",
        

    ];

    return (
        <div className="relative w-full min-h-[100vh] bg-gradient-to-br from-indigo-900 via-violet-950 to-black overflow-hidden">
          

            <div className="absolute inset-0 overflow-hidden z-0 opacity-60">
                <div className="flex w-[400%] animate-scroll">
                    {[...posterUrls, ...posterUrls].map((url, idx) => (
                        <div key={idx} className="w-[20%] h-screen">
                            <img
                                src={url}
                                alt="Movie Poster"
                                className="w-full h-full object-cover brightness-85"
                            />
                        </div>
                    ))}
                </div>
            </div>


          
            <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-black/60 to-black/90 z-10"></div>
            
           
            
            <div className="relative z-20 flex flex-col items-center justify-center text-center text-white px-6 pt-24 md:pt-60">
                
                <div className="backdrop-blur-md bg-white/10 border border-white/10 rounded-2xl p-10 max-w-3xl">
                    <h1 className="text-4xl md:text-4xl font-extrabold mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 text-transparent bg-clip-text drop-shadow-lg">
                        Discover Your Next Favorite Movie
                    </h1>
                    <p className="text-lg md:text-xl text-gray-300 mb-6">
                        MovieMate recommends films tailored to your tastes using smart AI. Dive in and explore a universe of cinema.
                    </p>
                    <a
                        href="#recommend"
                        className="bg-pink-600 text-white hover:bg-white hover:text-pink-600 px-8 py-3 rounded-lg text-lg font-semibold shadow-md transition"
                    >
                        Get Recommendations
                    </a>
                </div>
            

            </div>
        </div>
    );
};

export default Hero;
