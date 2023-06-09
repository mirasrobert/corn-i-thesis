import React, { useContext, useEffect, useRef } from "react";
import { Context } from "../Context";

const MalabananNorteSVG = () => {
	const { map, setHeatMap, setHeatMapItems } = useContext(Context);
	const ref = useRef(null);
	useEffect(() => {
		setHeatMap(ref.current);
		setHeatMapItems(ref.current.children);
	}, [ref, map, setHeatMap, setHeatMapItems]);
	return (
		<svg
			ref={ref}
			width='1152'
			height='1324'
			viewBox='0 0 1152 1324'
			id='malabanan_norte'
			className='mask'
			fill='none'
			xmlns='http://www.w3.org/2000/svg'>
			<path id="Rectangle 5" d="M144.316 16.6086L123.681 38.7533L96 94.1151V153H94.0349L88.5 306H553V153H555V0H249L177.533 7.04605L144.316 16.6086Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 5</textPath></text>
			</path>
			<path id="Rectangle 6" d="M708 0H555V153H553V306H1143V71.9167L1091.75 20.0435L1014 14V18H861V17.6151H766.885L708 0Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 6</textPath></text>
			</path >
			<path id="Rectangle 3" d="M86.1294 306L64 459H62.1214L38.5 612H37.4615L1 765H539V459H541V306H86.1294Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 3</textPath></text>
			</path >
			<path id="Rectangle 4" d="M541 306V459H539V765H1151V459H1153L1141.93 306H541Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 4</textPath></text>
			</path >
			<path id="Rectangle 1" d="M1 765V918H0.5L5.56211 1039.07L19.2298 1072H20L82.2897 1226H82.8222L165.043 1326H445.375V1292.84L541 1270.91V765H1Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 1</textPath></text>
			</path >
			<path id="Rectangle 2" d="M541 765V1267.18L557.609 1251.75L595.355 1267.18V1290.33L694 1296.5V1300H1060.9L1153 1272.77V1150.52L1119.78 1072V1006.65L1153 953.967V765H541Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 2</textPath></text>
			</path >

		</svg >
	);
};

export default MalabananNorteSVG;
