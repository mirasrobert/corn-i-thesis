import React from "react";
import { Context } from "../Context";

export default function MasalukotISVG() {
	const { map, setHeatMap, setHeatMapItems } = React.useContext(Context);
	const ref = React.useRef(null);
	React.useEffect(() => {
		setHeatMap(ref.current);
		setHeatMapItems(ref.current.children);
	}, [ref, map, setHeatMap, setHeatMapItems]);
	return (
		<svg
			ref={ref}
			width='582'
			height='1376'
			viewBox='0 0 582 1376'
			fill='none'
			id='masalukot_I'
			className='mask'
			xmlns='http://www.w3.org/2000/svg'>
			<path id="Rectangle 1" d="M244.165 83L260.532 0H585L552.667 166H551.805L519.667 331H197L229.139 166H228L244.367 83H244.165Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 1</textPath></text>
			</path>
			<path id="Rectangle 2" d="M366.409 331L366.604 330H522L512.007 414.005H511.74L496.035 496H498.5L482.166 580.005H482.152L465.917 663.5H303L303.292 662H141L157.334 578.5H157.249L173.192 497H172L188.25 413.491L198 331H366.409Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 2</textPath></text>
			</path >
			<path id="Rectangle 3" fill-rule="evenodd" clip-rule="evenodd" d="M140.546 662L124.304 745.498H123.282L107.039 829H106.622L90.3905 912.495H87.3301L71 996.5H233.878L233.975 996H395.667L411.806 913H414.667L430.708 830.5H431.904L448.137 747.005H449.167L465.5 663H303.005L303.2 662H140.546ZM286.363 746.495H285.742L285.896 745.702H286.517L286.363 746.495Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 3</textPath></text>
			</path >
			<path id="Rectangle 4" d="M69.6667 995L53 1080H53.2745L37.3922 1161H34.6667L18 1246H345.333L361.608 1163H364.333L381 1078H380.725L397 995H69.6667Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 4</textPath></text>
			</path >
			<path id="Rectangle 5" d="M16.6667 1244L0 1325L21.5789 1367.5L71.5789 1376L72.0814 1373.67L118.693 1355L161.104 1335L161.474 1333.31L246.363 1325L246.536 1324.16L275.579 1314.5L323.474 1290.5L344 1244H16.6667Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 5</textPath></text>
			</path >

		</svg >
	);
}
