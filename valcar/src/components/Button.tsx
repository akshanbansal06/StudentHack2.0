// Creating a reusable button component.
import React from 'react';

interface ButtonProps {
    label : string;
    onClick: () => void;
} 

const Button: React.FC<ButtonProps> = ({label, onClick}) => {
    return(
        <button
            onClick = {onClick}
            className="px-6 py-2 bg-blue0 font-Font text-white rounded-xl font-bold shadow-lg hover:shadow-blue-400 hover:scale-105 hover:bg-blue1 transition duration-300 ease-in-out transform cursor-pointer ring-1 ring-blue-500/40 hover:ring-2"
        >
            {label}
        </button>
    );
};

export default Button;