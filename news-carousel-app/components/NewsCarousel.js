"use client";

import React, { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, Calendar, MapPin } from "lucide-react";
import { Playfair_Display } from "next/font/google";

// Load Playfair Display with weights and subsets you want
const playfair = Playfair_Display({
  subsets: ["latin"],
  weight: ["700"], // bold for extra impact
  style: ["normal"],
});

const NewsCarousel = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);

  // Sample news data
  const newsItems = [
    {
      id: 1,
      image:
        "https://img.newspapers.com/img/img?user=7797238&id=185495674&clippingId=42677354&width=820&height=769&crop=671_454_3156_2960&rotation=0",
    },
    {
      id: 2,
      image:
        "https://www.detroitnews.com/gcdn/presto/2021/11/30/PDTN/acf7f6b3-1cb4-466a-b06d-b3aecb01b266-school_shooting.JPG?width=1320&height=616&fit=crop&format=pjpg&auto=webp",
    },
    {
      id: 3,
      image:
        "https://s3.amazonaws.com/arc-wordpress-client-uploads/tbt/wp-content/uploads/2018/02/26152518/Columbine-19991.jpg",
    },
    {
      id: 4,
      image:
        "https://lh3.googleusercontent.com/proxy/M8VcT9P1FQDSApXMSrVVvRIDevt0Mi8XjoOOP3jfnFy4R2jgnXZHVmFq_fzpKNEN9gVVcinUwbWK5I38ApGPxpZtTv2JT5ksgEbOlhj_wAjVXQnPUfPDYNsAHyPr4zxsylscYDcrxY0_Q5IwkLBLECC4gCXJrnBFfUh1sGKh9fCWENcrfgW7kqWXOKp-Ett9eIb1YRz7x1WlBdt3KDsLK0sTyQ",
    },
  ];

  // Auto-play functionality
  useEffect(() => {
    if (!isAutoPlaying) return;

    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % newsItems.length);
    }, 4000);

    return () => clearInterval(interval);
  }, [isAutoPlaying, newsItems.length]);

  const goToSlide = (index) => {
    setCurrentIndex(index);
    setIsAutoPlaying(false);
    // Resume auto-play after 5 seconds of inactivity
    setTimeout(() => setIsAutoPlaying(true), 5000);
  };

  const nextSlide = () => {
    setCurrentIndex((prev) => (prev + 1) % newsItems.length);
  };

  const prevSlide = () => {
    setCurrentIndex((prev) => (prev - 1 + newsItems.length) % newsItems.length);
  };

  return (
    <div className="w-full max-w-6xl mx-auto bg-white rounded-2xl shadow-2xl overflow-hidden">
      <div className="flex h-96">
        {/* Left Slider Section */}
        <div className="flex-1 relative overflow-hidden">
          <div
            className="flex transition-transform duration-500 ease-in-out h-full"
            style={{ transform: `translateX(-${currentIndex * 100}%)` }}
          >
            {newsItems.map((item, index) => (
              <div key={item.id} className="w-full flex-shrink-0 relative">
                <div className="absolute inset-0 flex items-center justify-center z-20 px-4">
                  <h2
                    className={`${playfair.className} text-red-700 text-4xl md:text-5xl font-extrabold drop-shadow-lg text-center`}
                    style={{
                      lineHeight: 1.2,
                      textShadow:
                        "2px 2px 4px rgba(0,0,0,0.8), 0 0 8px rgba(255,0,0,0.6)",
                    }}                    
                  >
                    Too Many Tragedies
                  </h2>
                </div>
                <div className="absolute inset-0 bg-gradient-to-r from-black/70 via-black/40 to-transparent z-10" />
                <img
                  src={item.image}
                  className="w-full relative"
                  style={{
                    position: "absolute",
                    top: "-100px", // controls vertical cropping of the image
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewsCarousel;
