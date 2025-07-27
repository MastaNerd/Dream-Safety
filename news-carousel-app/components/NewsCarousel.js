"use client";

import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Calendar, MapPin } from 'lucide-react';

const NewsCarousel = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);

  // Sample news data
  const newsItems = [
    {
      id: 1,
      title: "Pearl High Schol Shooting",
      description: "The Pearl High School shooting occurred on October 1, 1997, at Pearl High School in Pearl, Mississippi, United States.",
      image: "https://img.newspapers.com/img/img?user=7797238&id=185495674&clippingId=42677354&width=820&height=769&crop=671_454_3156_2960&rotation=0",
      date: "October 1st, 1997",
      location: "Pearl, MS",
    },
    {
      id: 2,
      title: "Bath School Bombing",
      description: "Disgruntled school board treasurer Andrew Kehoe killed 38 children and six adults with a homemade explosive and wounded many others.",
      image: "https://www.detroitnews.com/gcdn/presto/2021/11/30/PDTN/acf7f6b3-1cb4-466a-b06d-b3aecb01b266-school_shooting.JPG?width=1320&height=616&fit=crop&format=pjpg&auto=webp",
      date: "May 18, 1927",
      location: "Bath, MI",
    }
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

  const getCategoryColor = (category) => {
    const colors = {
      Technology: 'bg-blue-500',
      Environment: 'bg-green-500',
      Sports: 'bg-orange-500',
      Health: 'bg-red-500',
      Space: 'bg-purple-500'
    };
    return colors[category] || 'bg-gray-500';
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
              <div
                key={item.id}
                className="w-full flex-shrink-0 relative"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-black/70 via-black/40 to-transparent z-10" />
                <img
                  src={item.image}
                  alt={item.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-0 left-0 right-0 p-6 z-20 text-white">
                  <h2 className="text-2xl font-bold mb-2 line-clamp-2">
                    {item.title}
                  </h2>
                  <p className="text-gray-200 mb-3 line-clamp-2">
                    {item.description}
                  </p>
                  <div className="flex items-center gap-4 text-sm text-gray-300">
                    <div className="flex items-center gap-1">
                      <Calendar size={16} />
                      <span>{item.date}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin size={16} />
                      <span>{item.location}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Navigation Arrows */}
          <button
            onClick={prevSlide}
            className="absolute left-4 top-1/2 -translate-y-1/2 z-30 bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-full p-2 transition-all duration-200"
          >
            <ChevronLeft className="w-6 h-6 text-white" />
          </button>
          <button
            onClick={nextSlide}
            className="absolute right-4 top-1/2 -translate-y-1/2 z-30 bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-full p-2 transition-all duration-200"
          >
            <ChevronRight className="w-6 h-6 text-white" />
          </button>

          {/* Progress Dots */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 flex space-x-2">
            {newsItems.map((_, index) => (
              <button
                key={index}
                onClick={() => goToSlide(index)}
                className={`w-2 h-2 rounded-full transition-all duration-200 ${
                  index === currentIndex 
                    ? 'bg-white w-8' 
                    : 'bg-white/50 hover:bg-white/70'
                }`}
              />
            ))}
          </div>
        </div>

        {/* Right Quick Selection Panel */}
        <div className="w-72 bg-gray-50 border-l border-gray-200">
          <div className="p-4 border-b border-gray-200">
            <h3 className="font-bold text-gray-800 text-lg">Quick Select</h3>
            <p className="text-sm text-gray-600">Jump to any story</p>
          </div>
          <div className="p-2 space-y-2 max-h-80 overflow-y-auto">
            {newsItems.map((item, index) => (
              <button
                key={item.id}
                onClick={() => goToSlide(index)}
                className={`w-full p-3 rounded-lg text-left transition-all duration-200 hover:shadow-md ${
                  index === currentIndex
                    ? 'bg-white shadow-md border-l-4 border-blue-500'
                    : 'bg-white/70 hover:bg-white'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className="w-12 h-12 bg-gray-200 rounded-lg overflow-hidden flex-shrink-0">
                    <img
                      src={item.image}
                      alt={item.title}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className={`inline-block px-2 py-1 rounded text-xs font-medium text-white mb-1 ${getCategoryColor(item.category)}`}>
                      {item.category}
                    </div>
                    <h4 className="font-semibold text-sm text-gray-800 line-clamp-2 mb-1">
                      {item.title}
                    </h4>
                    <p className="text-xs text-gray-600">
                      {item.date}
                    </p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewsCarousel;