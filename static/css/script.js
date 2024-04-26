// 更新范围值的函数  
function updateValues() {  
    const minSlider = document.getElementById('minSlider'); // 最小滑块元素  
    const maxSlider = document.getElementById('maxSlider'); // 最大滑块元素  
    const minValue = document.getElementById('minValue'); // 显示最小值的元素  
    const maxValue = document.getElementById('maxValue'); // 显示最大值的元素  
    const rangeValues = document.querySelector('.range-values'); // 范围值显示元素选择器  
    const rangeValuesWidth = rangeValues.offsetWidth; // 范围值显示的宽度（像素）  
    const sliderRangeWidth = document.querySelector('.slider-range').offsetWidth; // 滑块范围的宽度（像素）  
    const minSliderPosition = minSlider.valueAsNumber / 100 * sliderRangeWidth; // 最小滑块当前位置（像素）根据滑块范围的宽度计算出百分比位置并转换为像素位置。  
    const maxSliderPosition = maxSlider.valueAsNumber / 100 * sliderRangeWidth; // 最大滑块当前位置（像素）根据滑块范围的宽度计算出百分比位置并转换为像素位置。  
    const minValueWidth = minValue.offsetWidth; // 最小值显示的宽度（像素）用于计算正确的偏移量以保持范围值显示居中。由于范围值显示宽度是固定的，因此需要除以范围值显示的宽度并乘以100来计算正确的百分比偏移量。同样的计算也应用于最大值显示。