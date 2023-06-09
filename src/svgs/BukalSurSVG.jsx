import React, { useContext, useEffect, useRef } from "react";
import { Context } from "../Context";

const BukalSurSVG = () => {
	const { map, setHeatMap, setHeatMapItems } = useContext(Context);
	const ref = useRef(null);
	useEffect(() => {
		setHeatMap(ref.current);
		setHeatMapItems(ref.current.children);
	}, [ref, map, setHeatMap, setHeatMapItems]);
	return (
		<svg
			ref={ref}
			width='1909'
			height='579'
			viewBox='0 0 1909 579'
			fill='none'
			id='bukal_sur'
			className='mask'
			xmlns='http://www.w3.org/2000/svg'>
			<path id="Rectangle 1" d="M102.242 37L31.1825 55.15L14.0476 92.4583L8 158H8.05618L0.5 242.7L8.05618 279H9.5L23.1084 298.192L76.0301 333.545L108.791 360.818L135 379V382.103L165.754 422.841L215.163 441.45L256 450V452.594L326.079 471.306L358.85 481.42L377 488.5V489.781L498 515V37H102.242Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 1</textPath></text>
			</path>
			<path id="Rectangle 2" d="M876.629 4.06667L861 20.8417V22.5L832.263 35.1887H740V36H498V514.623L619 541.5V540.196L740 565.5V566.525L771.258 581L861 566.525V566L982 552.158V0L876.629 4.06667Z" fill="#D9D9D9">

				<text><textPath className="tooltiptext">Crop Group 2</textPath></text>

			</path >
			<path id="Rectangle 3" d="M1103 10.1793L982 1.5V555.5L1103 545.138V544.5H1224V543.5H1345V540.5H1376.76L1389.37 530.037H1466V18H1345V10H1103V10.1793Z" fill="#D9D9D9">

				<text><textPath className="tooltiptext">Crop Group 3</textPath></text>

			</path >
			<path id="Rectangle 4" d="M1757.41 17H1708V18H1466V530.2H1587V528.2L1708 521.094V521.2H1739.26L1758.92 508.502H1829V508.2H1910L1875.07 387.116L1829 262.5V138H1828.53L1795.22 53.1508L1757.41 17Z" fill="#D9D9D9">

				<text><textPath className="tooltiptext">Crop Group 4</textPath></text>

			</path >

		</svg >
	);
};

export default BukalSurSVG;
